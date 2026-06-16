"""Split-graph SmolVLA export for the Orin Nano 8 GB (per-component TRT engines).

WHY (see orin-nano/smolvla-runtime/notes/findings.md): the monolithic
`sample_actions` export (export_valid_onnx.py) CANNOT TRT-build on the 8 GB Orin
Nano — TRT imports all 450M weights as FP32 working copies at once (~6 GB floor,
node-count-independent), so the build OOMs regardless of FP16 / --num-steps.

The validated fix is to split the model into per-component graphs, each carrying
only its weight slice (so each builds in ≤60 s), and run the flow-matching denoise
loop in Python (prefill ×1 -> decode ×N). This mirrors HF `ainekko/smolvla_base_onnx`
+ github.com/aifoundry-org/ETARS, adapted to our lerobot 0.5.1 weights/config.

Graphs written (9), single RGB image to match the Orin runtime:
  smolvlm_vision      image[1,3,512,512]            -> img_embeds[1,64,960]
  smolvlm_text        tokens[1,T]                   -> lang_embeds[1,T,960]   (dynamic T)
  smolvlm_expert_prefill  (mask,pos,vlm_embeds)     -> 32 KV tensors (fill_kv_cache)
  smolvlm_expert_decode   (mask,pos,expert_embeds,*KV) -> expert_out[1,50,720]
  state_projector     state[1,32]                   -> [1,960]
  action_in_projector action[1,50,32]               -> [1,50,720]
  action_out_projector expert_out[1,50,720]         -> v_t[1,50,32]
  time_in_projector   action_time[1,50,1440]        -> [1,50,H]
  time_out_projector  [1,50,H]                       -> [1,50,720]

The monolithic export stays as the FP32 parity gold (run parity on a big box).
Deploy bundle = these 9 graphs + tokenizer/ + normalization stats; the Orin builds
one FP16 TRT engine per heavy graph and runs the loop in backends/ort.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

# Reuse the legacy-ONNX-export monkeypatches the monolith needed (torch.where for
# boolean-mask NonZero, etc.) — the expert/vision graphs hit the same paths.
from export_valid_onnx import patch_smolvla_for_legacy_onnx_export

OPSET = 17


# --- component wrappers (mirror ETARS smolVLA_export.ipynb) -------------------
class VisionWrap(nn.Module):
    def __init__(self, vlme):
        super().__init__()
        self.vlme = vlme

    def forward(self, image):
        return self.vlme.embed_image(image)


class TextWrap(nn.Module):
    def __init__(self, vlme):
        super().__init__()
        self.vlme = vlme

    def forward(self, tokens):
        return self.vlme.embed_language_tokens(tokens)


class PrefillWrap(nn.Module):
    """Prefix (image+text+state embeds) -> the VLM KV cache only.

    sample_actions discards the prefill's hidden output and keeps just the cache,
    so we emit the 32 KV tensors (16 layers x key/value) and nothing else.
    """

    def __init__(self, vlme):
        super().__init__()
        self.vlme = vlme
        self.num_vlm_layers = vlme.num_vlm_layers

    def forward(self, attention_mask, position_ids, vlm_embeds):
        _, new_kv = self.vlme.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[vlm_embeds, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        flat = []
        for i in range(self.num_vlm_layers):
            flat.append(new_kv[i]["key_states"])
            flat.append(new_kv[i]["value_states"])
        return tuple(flat)


class DecodeWrap(nn.Module):
    """One flow-matching step: (suffix embeds + prefill KV) -> expert hidden.

    The denoise loop reuses the prefill KV every step (decode does not update it),
    so we take the cache as input and emit only the expert hidden output.
    """

    def __init__(self, vlme):
        super().__init__()
        self.vlme = vlme
        self.num_vlm_layers = vlme.num_vlm_layers

    def forward(self, attention_mask, position_ids, expert_embeds, *past_kv_flat):
        past = {
            i // 2: {
                "key_states": past_kv_flat[i],
                "value_states": past_kv_flat[i + 1],
            }
            for i in range(0, len(past_kv_flat), 2)
        }
        embeds, _ = self.vlme.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past,
            inputs_embeds=[None, expert_embeds],
            use_cache=True,
            fill_kv_cache=False,
        )
        return embeds[1]  # expert hidden [1, 50, 720]


# --- vision Gather-indices int64 patch (ETARS) -------------------------------
def patch_gather_indices_int64(onnx_path: str) -> None:
    """Cast non-int Gather indices to int64 in place (TRT/ORT reject float idx)."""
    import onnx
    from onnx import TensorProto, helper, shape_inference

    m = onnx.load(onnx_path)
    try:
        m = shape_inference.infer_shapes(m)
    except Exception:
        pass
    g = m.graph
    int_ok = {TensorProto.INT64, TensorProto.INT32}
    vtype = {}
    for vi in list(g.input) + list(g.value_info) + list(g.output):
        if vi.type.tensor_type and vi.type.tensor_type.elem_type:
            vtype[vi.name] = vi.type.tensor_type.elem_type
    for init in g.initializer:
        vtype[init.name] = init.data_type
    consumers: dict[str, list[int]] = {}
    for i, n in enumerate(g.node):
        for inp in n.input:
            consumers.setdefault(inp, []).append(i)
    casted: dict[str, str] = {}
    changed = 0
    for n in list(g.node):
        if n.op_type in {"Gather", "GatherND", "GatherElements"} and len(n.input) > 1:
            idx = n.input[1]
            if vtype.get(idx) in int_ok:
                continue
            if idx in casted:
                n.input[1] = casted[idx]
                continue
            out = idx + "_idx_i64"
            cast = helper.make_node("Cast", [idx], [out], to=TensorProto.INT64,
                                    name=f"{idx}_to_i64")
            g.node.insert(min(consumers.get(idx, [0])), cast)
            casted[idx] = out
            vtype[out] = TensorProto.INT64
            n.input[1] = out
            changed += 1
    onnx.checker.check_model(m)
    # Re-save with external weights (the torch.onnx export already wrote a sibling
    # .onnx.data; keep the bundle format consistent — small graph + one weights file).
    data_name = Path(onnx_path).name + ".data"
    for stale in (onnx_path, onnx_path + ".data"):
        if Path(stale).exists():
            Path(stale).unlink()
    onnx.save(m, onnx_path, save_as_external_data=True,
              all_tensors_to_one_file=True, location=data_name, size_threshold=1024)
    print(f"  vision Gather int64 patch: cast {changed} indices (re-saved external)")


def _export(model, args_tuple, path, input_names, output_names, dynamic_axes=None):
    model.eval()
    torch.onnx.export(
        model, args_tuple, path,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=OPSET, do_constant_folding=False,
    )
    import onnx
    onnx.checker.check_model(path)
    print(f"  wrote {path}  ({Path(path).stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="lerobot/smolvla_base")
    ap.add_argument("--out-dir", default="exports-split")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    patch_smolvla_for_legacy_onnx_export()
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)

    print(f"Loading {args.model_id} on {dev} ...")
    policy = SmolVLAPolicy.from_pretrained(args.model_id).eval().float().to(dev)
    for p in policy.parameters():
        p.requires_grad_(False)
    m = policy.model
    vlme = m.vlm_with_expert
    cfg = policy.config

    L = vlme.num_vlm_layers
    vlm_dim = vlme.config.text_config.hidden_size            # 960
    exp_dim = vlme.expert_hidden_size                        # 720
    chunk = cfg.chunk_size                                   # 50
    act_dim = cfg.max_action_dim                             # 32
    state_dim = cfg.max_state_dim                            # 32
    img_h, img_w = cfg.resize_imgs_with_padding

    with torch.no_grad():
        # derive the real prefix length from a single image + padded lang + state
        n_img = vlme.embed_image(torch.zeros(1, 3, img_h, img_w, device=dev)).shape[1]
        lang_len = cfg.tokenizer_max_length
        prefix_len = n_img + lang_len + 1                    # +1 state token = 113
        print(f"dims: L={L} vlm_dim={vlm_dim} exp_dim={exp_dim} prefix_len={prefix_len} "
              f"chunk={chunk} act_dim={act_dim}")

        # 1) vision
        _export(VisionWrap(vlme), torch.zeros(1, 3, img_h, img_w, device=dev),
                str(out / "smolvlm_vision.onnx"), ["image"], ["img_embeds"])
        patch_gather_indices_int64(str(out / "smolvlm_vision.onnx"))

        # 2) text (dynamic sequence length)
        _export(TextWrap(vlme), torch.ones(1, lang_len, dtype=torch.long, device=dev),
                str(out / "smolvlm_text.onnx"), ["tokens"], ["lang_embeds"],
                dynamic_axes={"tokens": {1: "T"}, "lang_embeds": {1: "T"}})

        # 3) expert prefill -> KV cache (static prefix_len)
        kv_names = [f"{k}_{i}" for i in range(L) for k in ("present_key", "present_value")]
        pf_mask = torch.ones(1, prefix_len, prefix_len, dtype=torch.bool, device=dev)
        pf_pos = torch.arange(prefix_len, device=dev).unsqueeze(0)
        pf_emb = torch.randn(1, prefix_len, vlm_dim, device=dev)
        _export(PrefillWrap(vlme), (pf_mask, pf_pos, pf_emb),
                str(out / "smolvlm_expert_prefill.onnx"),
                ["attention_mask", "position_ids", "vlm_embeds"], kv_names)

        # grab the real KV shapes from a prefill run to build the decode dummies
        past_kv = PrefillWrap(vlme).forward(pf_mask, pf_pos, pf_emb)
        print(f"  KV[0] shape: {tuple(past_kv[0].shape)}  ({len(past_kv)} tensors)")

        # 4) expert decode (suffix attends to prefix KV + itself); total = prefix+chunk
        total = prefix_len + chunk
        dc_mask = torch.ones(1, chunk, total, dtype=torch.bool, device=dev)
        dc_pos = torch.arange(chunk, device=dev).unsqueeze(0)
        dc_emb = torch.randn(1, chunk, exp_dim, device=dev)
        in_names = ["attention_mask", "position_ids", "expert_embeds"] + \
                   [f"{k}_{i}" for i in range(L) for k in ("past_key", "past_value")]
        _export(DecodeWrap(vlme), (dc_mask, dc_pos, dc_emb, *past_kv),
                str(out / "smolvlm_expert_decode.onnx"), in_names, ["expert_out"])

        # 5-9) projectors (small, run-once to get shapes)
        def proj(mod, dummy, name, in_name="input"):
            o = mod(dummy)
            _export(mod, dummy, str(out / name), [in_name], ["output"])
            return o

        proj(m.state_proj, torch.zeros(1, state_dim, device=dev),
             "state_projector.onnx", "state")
        proj(m.action_in_proj, torch.zeros(1, chunk, act_dim, device=dev),
             "action_in_projector.onnx", "action")
        proj(m.action_out_proj, torch.zeros(1, chunk, exp_dim, device=dev),
             "action_out_projector.onnx", "expert_out")
        # time MLPs: in takes concat[action_emb, time_emb] = 2*exp_dim
        ti_out = proj(m.action_time_mlp_in, torch.zeros(1, chunk, 2 * exp_dim, device=dev),
                      "time_in_projector.onnx", "action_time")
        proj(m.action_time_mlp_out, torch.zeros(1, chunk, ti_out.shape[-1], device=dev),
             "time_out_projector.onnx", "hidden")

    # deploy bundle: tokenizer + normalization stats
    try:
        vlme.processor.tokenizer.save_pretrained(str(out / "tokenizer"))
        print(f"Saved tokenizer -> {out / 'tokenizer'}")
    except Exception as e:
        print(f"tokenizer save skipped: {e}")

    print(f"\nDONE — 9 split graphs in {out}/  (deploy bundle; FP32 gold stays the monolith)")


if __name__ == "__main__":
    main()
