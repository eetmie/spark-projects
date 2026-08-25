#!/usr/bin/env python3
"""Export X-VLA as split ONNX graphs sized for the 8 GB Orin.

See ../notes/split_design.md for the measured build-memory curve that sets the cut sizes.
Short version: TRT's build peak is ~3.18 GB + 5.63 x (FP32 weight GB), so an engine must
carry at most ~0.40 GB of weights (~100 M params) to build with real margin. X-VLA's three
heavy components are all over that on their own, so each is split by parameter budget:

    vision (1.44 GB)  DaViT convs/blocks, flattened and packed; last chunk adds the projector
    text   (0.83 GB)  BART embedding + 12 encoder layers
    denoise (1.21 GB) 24 transformer blocks, run once per denoising step

`generate_actions` runs the VLM once and the denoiser `num_denoising_steps` times, so the
loop-invariant conditioning projections are hoisted into their own `cond` graph -- exact,
not an approximation, because the sequence positions are fixed.

Each graph is exported in its own subprocess: loading the policy costs ~3.5 GB on CPU and
holding that alongside an export trace is what pushes this board into swap.

    python tools/export_split_onnx.py --checkpoint models/xvla-base --domain-id 0
    python tools/export_split_onnx.py --random-init --graphs vision   # smoke-test only
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

GRAPHS = ("vision", "text_encoder", "cond", "denoise")

# Per-engine FP32 weight budget, from tools/build_probe.py on this board. 0.40 GB built
# with 1.69 GB of headroom; 0.60 GB finished with only 0.47 GB left.
DEFAULT_BUDGET_GB = 0.40


def _vlm_parts(vlm):
    """(owner, vision_tower, multi_modal_projector, language_model) across transformers versions.

    transformers < 5 hangs the vision tower, the projector and the language model
    directly off `Florence2ForConditionalGeneration`. transformers 5.x moved them one
    level down onto a `Florence2Model` at `.model`, and only `get_image_features`
    survives as a forwarding method — so `vlm.multi_modal_projector` raises
    AttributeError there. Resolve through whichever level actually owns the modules,
    and return that owner so `del owner.vision_tower` frees the right reference.
    """
    owner = getattr(vlm, "model", None)
    if owner is None or not hasattr(owner, "language_model"):
        owner = vlm
    # getattr with a default, not attribute access: the text_encoder branch DELETES
    # vision_tower to free 1.4 GB before tracing, and then builds TextHead, which calls
    # back in here. Requiring the tower would make the second call fail on the first
    # call's cleanup.
    return (owner,
            getattr(owner, "vision_tower", None),
            getattr(owner, "multi_modal_projector", None),
            owner.language_model)


def _n_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def _validate_onnx(path, stem: str) -> None:
    """Reject a malformed graph at export time rather than at engine-build time.

    The TorchScript exporter can silently emit nodes whose *required* inputs are empty
    strings; the file writes fine and only fails minutes later when ORT tries to load it
    ("input 0 is marked single but has an empty string"). Checking here keeps that
    failure next to the code that caused it.

    An empty string is only a defect in a REQUIRED position. ONNX also uses "" as the
    legal way to omit a trailing optional input, which is exactly what the tracer emits
    for the DaViT window-attention `Pad` (`constant_value` omitted, because the pads are
    all zero at 224x224 -- every feature map divides by the window size). Flagging those
    rejected a graph that `onnx.checker` passes and that ORT loads happily, so the
    op schema decides which positions matter rather than a blanket rule.
    """
    import onnx
    from onnx import defs

    model = onnx.load(str(path), load_external_data=False)
    opset = {o.domain: o.version for o in model.opset_import}

    def required_positions(op_type: str, domain: str) -> set[int] | None:
        """Indices whose input must be present, or None when the schema is unknown."""
        try:
            schema = defs.get_schema(op_type, opset.get(domain, opset.get("", 17)),
                                     domain)
        except Exception:
            return None
        return {i for i, inp in enumerate(schema.inputs)
                if inp.option != defs.OpSchema.FormalParameterOption.Optional}

    dangling = []
    for n in model.graph.node:
        req = required_positions(n.op_type, n.domain)
        for i, name in enumerate(n.input):
            if name != "":
                continue
            # Unknown schema: fall back to the old blanket rule rather than pass it.
            if req is None or i in req:
                dangling.append((n.op_type, n.name, i))
    if dangling:
        raise RuntimeError(
            f"{stem}: {len(dangling)} node(s) exported with an empty REQUIRED input, "
            f"e.g. {dangling[:3]} -- the graph is unloadable. Usually a traced construct "
            f"the exporter mishandled (in-place indexed assignment, expand(-1, ...))."
        )
    onnx.checker.check_model(model)


def _plan_chunks(units: list[tuple[str, object]], budget_params: int) -> list[list[int]]:
    """Greedy contiguous packing of (name, module) units into chunks under the budget.

    Contiguous because these are sequential stacks: a chunk must be a run of adjacent
    layers for the intermediate tensor to be the only thing crossing the boundary.
    """
    chunks: list[list[int]] = []
    current: list[int] = []
    running = 0
    for i, (_, mod) in enumerate(units):
        n = _n_params(mod)
        if current and running + n > budget_params:
            chunks.append(current)
            current, running = [], 0
        current.append(i)
        running += n
    if current:
        chunks.append(current)
    return chunks


# ======================================================================================
# graph wrappers
# ======================================================================================


def _build_wrappers():
    """Imported lazily inside the child process so the parent never loads torch."""
    import torch
    from torch import nn

    from lerobot.policies.xvla.soft_transformer import timestep_embedding

    class SeqChunk(nn.Module):
        """A run of DaViT convs/blocks; the final chunk also applies the projector."""

        def __init__(self, mods, projector=None):
            super().__init__()
            self.mods = nn.ModuleList(mods)
            self.projector = projector

        def forward(self, x):
            for m in self.mods:
                x = m(x)
            if self.projector is not None:
                x = self.projector(x)
            return x

    class TextHead(nn.Module):
        """Token embedding + BART preamble + the first run of encoder layers.

        `BartEncoder.forward` cannot be reused here: it unconditionally calls
        `create_bidirectional_mask`, which does not survive JIT tracing (it reaches
        `sdpa_mask` with a tuple where it expects a tensor, raising IndexError), and
        passing `attention_mask=None` does not skip that call. So the preamble is
        reproduced explicitly -- it is exactly the four lines below, taken from
        `BartEncoder.forward`, minus the mask construction and the eval-time no-op
        dropout.

        Dropping the mask is not a simplification of the model: `forward_vlm` builds an
        all-ones mask over image+language tokens (it does not mask language padding), so
        every token attends to every token, which is what `attn_mask=None` means.
        `parity.py` checks that equivalence numerically against the reference.
        """

        def __init__(self, vlm, keep_layers: int):
            super().__init__()
            self.embed = vlm.get_input_embeddings()
            encoder = _vlm_parts(vlm)[3].encoder
            self.embed_positions = encoder.embed_positions
            self.layernorm_embedding = encoder.layernorm_embedding
            self.layers = nn.ModuleList(list(encoder.layers[:keep_layers]))

        def forward(self, input_ids, image_tokens):
            inputs_embeds = self.embed(input_ids)
            merged = torch.cat([image_tokens, inputs_embeds], dim=1)
            hidden = merged + self.embed_positions(merged[:, :, -1])
            hidden = self.layernorm_embedding(hidden)
            for layer in self.layers:
                hidden = layer(hidden, None)
            return hidden

    class TextChunk(nn.Module):
        """Remaining BART encoder layers.

        `attention_mask=None` is correct here, not a shortcut: `forward_vlm` builds an
        all-ones mask over image+language tokens (it does not mask language padding), so
        every token attends to every token.
        """

        def __init__(self, layers):
            super().__init__()
            self.layers = nn.ModuleList(layers)

        def forward(self, hidden):
            for layer in self.layers:
                hidden = layer(hidden, None)
            return hidden

    class CondGraph(nn.Module):
        """The hoisted loop-invariant half of `SoftPromptedTransformer.forward`.

        vlm_features [1,T_v,D], aux_visual [1,T_a,D] -> cond_tokens [1, T_v+T_a, H],
        already carrying their `pos_emb` slice. Positions are fixed, so folding the slice
        in here is equivalent to adding it inside the denoising loop.
        """

        def __init__(self, transformer, n_action: int):
            super().__init__()
            self.vlm_proj = transformer.vlm_proj
            self.aux_visual_proj = transformer.aux_visual_proj
            self.register_buffer("pos_emb", transformer.pos_emb.detach().clone())
            self.n_action = n_action

        def forward(self, vlm_features, aux_visual):
            cond = torch.cat(
                [self.vlm_proj(vlm_features), self.aux_visual_proj(aux_visual)], dim=1
            )
            return cond + self.pos_emb[:, self.n_action : self.n_action + cond.shape[1]]

    class DenoiseGraph(nn.Module):
        """One slice of the hot path, run once per denoising step.

        `first` slices take (x_t, t, proprio, cond_tokens) and assemble the sequence;
        `last` slices apply the final norm + action decoder. `domain_id` is baked: the
        DomainAwareLinear weights and the soft prompts are gathered at export time and
        become constants, so the 30-domain tables never enter an engine.
        """

        def __init__(self, transformer, action_space, blocks, *, first, last, domain_id,
                     dim_time, n_action, dim_action=None, dim_proprio=None):
            super().__init__()
            self.blocks = blocks
            self.first = first
            self.last = last
            self.dim_time = dim_time
            self.n_action = n_action

            did = torch.tensor([domain_id], dtype=torch.long)
            if first:
                # EE6DActionSpace.preprocess zeroes the gripper channels. Done with
                # constant masks built here, in eager mode, rather than `ones_like` plus
                # an indexed assignment inside forward: the TorchScript exporter mistraced
                # that into Add/Expand nodes with empty input names, producing an ONNX
                # file that loads nowhere ("input 0 is marked single but has an empty
                # string"). Shapes are static, so a constant mask is equivalent.
                gripper_idx = list(action_space.gripper_idx)
                amask = torch.ones(1, 1, dim_action)
                amask[..., gripper_idx] = 0.0
                self.register_buffer("action_mask", amask)
                pmask = torch.ones(1, dim_proprio)
                pmask[..., gripper_idx] = 0.0
                self.register_buffer("proprio_mask", pmask)

                enc = transformer.action_encoder
                self.register_buffer(
                    "enc_w", enc.fc(did).view(enc.input_size, enc.output_size).detach().clone()
                )
                self.register_buffer("enc_b", enc.bias(did).view(1, 1, -1).detach().clone())
                self.register_buffer(
                    "pos_emb_action", transformer.pos_emb[:, :n_action].detach().clone()
                )
                self.register_buffer(
                    "soft_prompts",
                    transformer.soft_prompt_hub(did)
                    .view(1, transformer.len_soft_prompts, transformer.hidden_size)
                    .detach()
                    .clone(),
                )
            if last:
                dec = transformer.action_decoder
                self.norm = transformer.norm
                self.register_buffer(
                    "dec_w", dec.fc(did).view(dec.input_size, dec.output_size).detach().clone()
                )
                self.register_buffer("dec_b", dec.bias(did).view(1, 1, -1).detach().clone())

        def forward(self, *args):
            if self.first:
                x_t, t, proprio, cond_tokens = args
                x_t = x_t * self.action_mask
                proprio = proprio * self.proprio_mask

                # `repeat` with static counts, not `expand(-1, n, -1)`: the -1s traced
                # into Expand nodes with empty inputs (see the mask comment above).
                time_emb = timestep_embedding(t, self.dim_time)
                time_tokens = time_emb.unsqueeze(1).repeat(1, self.n_action, 1)
                proprio_tokens = proprio.unsqueeze(1).repeat(1, self.n_action, 1)
                tokens = torch.cat([x_t, proprio_tokens, time_tokens], dim=-1)

                x = torch.matmul(tokens, self.enc_w) + self.enc_b
                x = x + self.pos_emb_action
                x = torch.cat([x, cond_tokens, self.soft_prompts], dim=1)
            else:
                (x,) = args

            for block in self.blocks:
                x = block(x)

            if not self.last:
                return x
            head = self.norm(x[:, : self.n_action])
            return torch.matmul(head, self.dec_w) + self.dec_b

    return SeqChunk, TextHead, TextChunk, CondGraph, DenoiseGraph


# ======================================================================================
# child: export one graph family
# ======================================================================================


def export_one(args) -> None:
    import gc

    import torch
    from torch import nn

    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from lerobot.configs.policies import PreTrainedConfig

    # The checkpoint's config says device=cuda, and both constructors honour it. Export
    # on CPU: the trace samples are CPU tensors, and 3.5 GB of weights on an Orin's
    # unified memory would come straight out of the same budget the build needs.
    cfg_obj = PreTrainedConfig.from_pretrained(str(args.checkpoint))
    cfg_obj.device = "cpu"

    if args.random_init:
        # Exercises the export path (tracing, opset coverage, shapes) without the 3.5 GB
        # checkpoint. Topology and shapes are identical; only values differ, so this
        # catches "DaViT will not trace" long before the weights land.
        policy = XVLAPolicy(cfg_obj)
    else:
        policy = XVLAPolicy.from_pretrained(str(args.checkpoint), config=cfg_obj)
    policy.to("cpu").eval()
    model = policy.model
    cfg = policy.config

    SeqChunk, TextHead, TextChunk, CondGraph, DenoiseGraph = _build_wrappers()

    hidden = cfg.hidden_size
    n_views = cfg.num_image_views
    img_hw = tuple(cfg.resize_imgs_with_padding or (224, 224))
    lang_len = args.lang_len
    proj_dim = model.vlm.config.vision_config.projection_dim
    budget = int(args.budget_gb * 1e9 / 4)

    meta: dict = {}
    exported: list[dict] = []

    def dump(module, sample, names_in, names_out, stem):
        path = out_dir / f"{stem}.onnx"
        module = module.eval().float()
        with torch.no_grad():
            torch.onnx.export(
                module, sample, str(path),
                input_names=names_in, output_names=names_out,
                opset_version=args.opset, dynamo=False,
            )
        _validate_onnx(path, stem)
        exported.append({
            "name": stem,
            "file": path.name,
            "inputs": names_in,
            "outputs": names_out,
            "params": _n_params(module),
            "size_mb": round(path.stat().st_size / 1e6, 1),
        })
        print(f"  {stem}: {_n_params(module) / 1e6:.1f}M params, "
              f"{path.stat().st_size / 1e6:.0f} MB", flush=True)

    # -- vision ------------------------------------------------------------------------
    if args.graph == "vision":
        _, tower, projector, _ = _vlm_parts(model.vlm)
        units: list[tuple[str, object]] = []
        for si, (conv, block) in enumerate(zip(tower.convs, tower.blocks)):
            units.append((f"conv{si}", conv))
            for li, layer in enumerate(block):
                units.append((f"stage{si}.layer{li}", layer))

        plan = _plan_chunks(units, budget)
        print(f"  vision: {len(units)} units -> {len(plan)} engines "
              f"(budget {args.budget_gb:.2f} GB)")

        x = torch.zeros(args.valid_views, 3, *img_hw)
        with torch.no_grad():
            for ci, idxs in enumerate(plan):
                last = ci == len(plan) - 1
                chunk = SeqChunk(
                    [units[i][1] for i in idxs], projector=projector if last else None
                )
                stem = "vision" if len(plan) == 1 else f"vision_{ci}"
                name_in = "pixel_values" if ci == 0 else "hidden_in"
                name_out = "image_features" if last else "hidden_out"
                dump(chunk, (x,), [name_in], [name_out], stem)
                x = chunk(x)
        meta["tokens_per_view"] = int(x.shape[1])

    # -- text encoder ------------------------------------------------------------------
    elif args.graph == "text_encoder":
        _owner, _, _, _lm = _vlm_parts(model.vlm)
        del _owner.vision_tower
        gc.collect()
        vlm = model.vlm
        layers = list(_lm.encoder.layers)
        embed_params = _n_params(vlm.get_input_embeddings())

        # The embedding table rides with the head chunk, so it eats into that budget.
        units = [(f"layer{i}", layer) for i, layer in enumerate(layers)]
        head_budget = max(budget - embed_params, _n_params(layers[0]))
        first = _plan_chunks(units, head_budget)[0]
        rest = _plan_chunks(units[len(first):], budget)
        print(f"  text: embed {embed_params / 1e6:.1f}M + {len(layers)} layers -> "
              f"{1 + len(rest)} engines")

        ids = torch.zeros(1, lang_len, dtype=torch.long)
        img = torch.zeros(1, args.tokens_per_view, proj_dim)
        head = TextHead(vlm, keep_layers=len(first))
        stem = "text_encoder" if not rest else "text_encoder_0"
        dump(head, (ids, img), ["input_ids", "image_tokens"],
             ["vlm_features" if not rest else "hidden_out"], stem)

        with torch.no_grad():
            h = head(ids, img)
        offset = len(first)
        for ci, idxs in enumerate(rest):
            chunk = TextChunk([layers[offset + i] for i in idxs])
            last = ci == len(rest) - 1
            dump(chunk, (h,), ["hidden_in"],
                 ["vlm_features" if last else "hidden_out"], f"text_encoder_{ci + 1}")
            with torch.no_grad():
                h = chunk(h)

    # -- conditioning ------------------------------------------------------------------
    elif args.graph == "cond":
        cg = CondGraph(model.transformer, cfg.chunk_size)
        del model.vlm
        gc.collect()
        t_v = args.tokens_per_view + lang_len
        t_a = (n_views - 1) * args.tokens_per_view
        dump(cg, (torch.zeros(1, t_v, proj_dim), torch.zeros(1, t_a, proj_dim)),
             ["vlm_features", "aux_visual"], ["cond_tokens"], "cond")

    # -- denoiser ----------------------------------------------------------------------
    elif args.graph == "denoise":
        del model.vlm
        gc.collect()
        tr = model.transformer
        splits = args.policy_splits
        if sum(splits) != cfg.depth:
            sys.exit(f"--policy-splits {splits} must sum to depth {cfg.depth}")

        t_v = args.tokens_per_view + lang_len
        t_a = (n_views - 1) * args.tokens_per_view
        n_cond = t_v + t_a
        seq_len = cfg.chunk_size + n_cond + cfg.len_soft_prompts
        meta["seq_len"] = seq_len
        print(f"  denoise: {cfg.depth} blocks -> {len(splits)} engines {splits}, "
              f"seq_len {seq_len}")

        start = 0
        for i, count in enumerate(splits):
            blocks = nn.ModuleList(list(tr.blocks[start : start + count]))
            start += count
            first_c, last_c = i == 0, i == len(splits) - 1
            dg = DenoiseGraph(
                tr, model.action_space, blocks,
                first=first_c, last=last_c, domain_id=args.domain_id,
                dim_time=cfg.dim_time, n_action=cfg.chunk_size,
                dim_action=model.dim_action, dim_proprio=model.dim_proprio,
            )
            stem = "denoise" if len(splits) == 1 else f"denoise_{i}"
            if first_c:
                sample = (
                    torch.zeros(1, cfg.chunk_size, model.dim_action),
                    torch.zeros(1),
                    torch.zeros(1, model.dim_proprio),
                    torch.zeros(1, n_cond, hidden),
                )
                names_in = ["x_t", "t", "proprio", "cond_tokens"]
            else:
                sample = (torch.zeros(1, seq_len, hidden),)
                names_in = ["hidden_in"]
            dump(dg, sample, names_in, ["action"] if last_c else ["hidden_out"], stem)

    (out_dir / f"_meta_{args.graph}.json").write_text(
        json.dumps({"graphs": exported, **meta}, indent=2)
    )


# ======================================================================================
# parent: orchestrate one subprocess per graph family
# ======================================================================================


def _provenance(checkpoint: Path, opset: int) -> dict:
    """Who built this bundle, from what, with which library versions.

    X-VLA's split is budget-driven: change the budget, the transformers version or the
    checkpoint and you get a different set of graphs that still loads. Without this
    block a bundle cannot be told apart from another bundle after the fact, which
    matters more here than on the SmolVLA side because this one is meant to be
    published and re-downloaded by people who did not build it.
    """
    import subprocess

    def _ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None

    try:
        sha = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent),
                              "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        sha = None

    return {
        "checkpoint": str(checkpoint),
        "exported_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "exporter_sha": sha,
        "opset": opset,
        "lerobot": _ver("lerobot"),
        "torch": _ver("torch"),
        "transformers": _ver("transformers"),
    }


def write_manifest(out: Path) -> int:
    """sha256 of every bundle file, so a transfer can be verified on arrival.

    Written last and excludes itself. Twelve graphs travel as .onnx + .onnx.data
    pairs; a truncated external-data file fails minutes later at engine-build time
    with an error that names neither the file nor the cause.
    """
    import hashlib

    lines = []
    for f in sorted(out.rglob("*")):
        # `_meta_*.json` are exporter scratch — one per graph family, used only to
        # rebuild bundle.json on a partial re-export — and ship_bundle.sh excludes
        # them from the transfer. A manifest must describe the bundle AS SHIPPED, or
        # verification on the target fails on files that were never meant to travel.
        if (not f.is_file() or f.name == "MANIFEST.sha256"
                or (f.name.startswith("_meta_") and f.suffix == ".json")):
            continue
        h = hashlib.sha256()
        with f.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        lines.append(f"{h.hexdigest()}  {f.relative_to(out)}")
    (out / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    return len(lines)



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=REPO / "models" / "xvla-base")
    ap.add_argument("--out-dir", type=Path, default=REPO / "exports" / "split")
    ap.add_argument("--domain-id", type=int, default=0,
                    help="baked into the action encoder/decoder and soft prompts")
    ap.add_argument("--valid-views", type=int, default=None,
                    help="static batch of the vision engine; defaults to num_image_views. "
                         "Set to the number of REAL cameras -- padded views are zeroed by "
                         "the runtime and never need a forward pass.")
    ap.add_argument("--lang-len", type=int, default=50,
                    help="language tokens; 50 comes from the checkpoint's policy_preprocessor.json")
    ap.add_argument("--budget-gb", type=float, default=DEFAULT_BUDGET_GB,
                    help="FP32 weight budget per engine (see notes/split_design.md)")
    ap.add_argument("--policy-splits", type=int, nargs="+", default=[6, 6, 6, 6],
                    help="block counts per denoise engine, must sum to depth (24)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--graphs", nargs="+", default=list(GRAPHS), choices=GRAPHS)
    ap.add_argument("--random-init", action="store_true",
                    help="build from config with random weights instead of loading the "
                         "checkpoint -- smoke-tests the export path only")
    ap.add_argument("--bundle-only", action="store_true",
                    help="rebuild bundle.json from the _meta_*.json already in --out-dir, "
                         "without exporting anything")
    ap.add_argument("--_graph", dest="graph", help=argparse.SUPPRESS)
    ap.add_argument("--tokens-per-view", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.graph:  # child invocation
        export_one(args)
        return

    cfg = json.loads((args.checkpoint / "config.json").read_text())
    if args.valid_views is None:
        args.valid_views = cfg.get("num_image_views") or 3
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"exporting X-VLA split graphs -> {args.out_dir}")
    print(f"  domain_id={args.domain_id}  valid_views={args.valid_views}  "
          f"lang_len={args.lang_len}  budget={args.budget_gb:.2f} GB  "
          f"policy_splits={args.policy_splits}\n")

    tokens_per_view = args.tokens_per_view
    for graph in [] if args.bundle_only else args.graphs:
        if graph != "vision" and tokens_per_view is None:
            sys.exit(f"'{graph}' needs tokens_per_view: export 'vision' first, or pass "
                     f"--tokens-per-view")
        print(f"[{graph}]", flush=True)
        cmd = [
            sys.executable, __file__, "--_graph", graph,
            "--checkpoint", str(args.checkpoint), "--out-dir", str(args.out_dir),
            "--domain-id", str(args.domain_id), "--valid-views", str(args.valid_views),
            "--lang-len", str(args.lang_len), "--opset", str(args.opset),
            "--budget-gb", str(args.budget_gb),
            "--policy-splits", *[str(s) for s in args.policy_splits],
        ]
        if tokens_per_view is not None:
            cmd += ["--tokens-per-view", str(tokens_per_view)]
        if args.random_init:
            cmd.append("--random-init")
        if subprocess.run(cmd, text=True).returncode != 0:
            sys.exit(f"export of '{graph}' failed")

        meta = json.loads((args.out_dir / f"_meta_{graph}.json").read_text())
        if "tokens_per_view" in meta:
            tokens_per_view = meta["tokens_per_view"]
            print(f"  tokens_per_view = {tokens_per_view}")

    # Collect from every _meta_*.json on disk, not just this run's --graphs, so a partial
    # re-export (e.g. redoing text_encoder alone) still writes a complete bundle.
    order = {name: i for i, name in enumerate(GRAPHS)}
    metas = sorted(
        args.out_dir.glob("_meta_*.json"),
        key=lambda p: order.get(p.stem[len("_meta_"):], len(order)),
    )
    graphs = [g for m in metas for g in json.loads(m.read_text())["graphs"]]
    missing = [n for n in GRAPHS if not (args.out_dir / f"_meta_{n}.json").exists()]
    if missing:
        print(f"  note: no graphs exported yet for {missing}")
    if tokens_per_view is None:
        vision_meta = args.out_dir / "_meta_vision.json"
        if vision_meta.exists():
            tokens_per_view = json.loads(vision_meta.read_text()).get("tokens_per_view")
    bundle = {
        "domain_id": args.domain_id,
        "valid_views": args.valid_views,
        "num_image_views": cfg.get("num_image_views"),
        "lang_len": args.lang_len,
        "tokens_per_view": tokens_per_view,
        "chunk_size": cfg.get("chunk_size"),
        "num_denoising_steps": cfg.get("num_denoising_steps"),
        "hidden_size": cfg.get("hidden_size"),
        "dim_time": cfg.get("dim_time"),
        "max_state_dim": cfg.get("max_state_dim"),
        "action_mode": cfg.get("action_mode"),
        "policy_splits": args.policy_splits,
        "budget_gb": args.budget_gb,
        "graphs": graphs,
        "provenance": _provenance(args.checkpoint, args.opset),
    }
    (args.out_dir / "bundle.json").write_text(json.dumps(bundle, indent=2))
    total = sum(g["params"] for g in graphs)
    print(f"\n{len(graphs)} graphs, {total / 1e6:.1f}M params total")
    print(f"largest engine: {max(g['params'] for g in graphs) * 4 / 1e9:.2f} GB fp32")
    print(f"wrote {args.out_dir / 'bundle.json'}")
    if missing:
        print("skipping MANIFEST.sha256 — the bundle is incomplete "
              f"(no graphs for {missing})")
    else:
        n = write_manifest(args.out_dir)
        print(f"wrote {args.out_dir / 'MANIFEST.sha256'} ({n} files)\n"
              f"  verify after transfer:  cd <bundle> && sha256sum -c MANIFEST.sha256")


if __name__ == "__main__":
    main()
