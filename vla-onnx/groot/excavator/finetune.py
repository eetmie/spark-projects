"""NVIDIA's GR00T N1.7 fine-tune (gr00t/experiment/launch_finetune.py), run unchanged, with the
excavator modality config and optional training-time RTC.

    python finetune.py [--rtc-max-delay N] [--rtc-delay-dist uniform|exp] -- LAUNCH_FINETUNE_ARGS...

`--rtc-max-delay 0` (default) trains the stock loss. Anything above installs rtc_patch.py
and writes OUTPUT_DIR/rtc_training.json; the exporter and runtime need it to know the
checkpoint expects per-row time. `--modality-config-path` defaults to modality.py here and
`--embodiment-tag` to NEW_EMBODIMENT.

The backbone constructor calls Qwen3VLForConditionalGeneration.from_pretrained(Cosmos-Reason2-2B),
which downloads 2B weights only for the GR00T checkpoint to overwrite them. By default the VLM
is built from its config instead (as groot17_split.load_policy does); only the gated repo's
config/processor files are needed. `--vlm-weights download` restores the stock behavior.

Both gated repos (nvidia/GR00T-N1.7-3B, nvidia/Cosmos-Reason2-2B) resolve to their local HF
cache snapshot when fully cached, so a run needs no Hugging Face login. The checkpoint still
records the repo ids.
"""
import argparse
import pathlib
import runpy
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rtc_patch  # noqa: E402
from gr00t.experiment import launch_finetune  # noqa: E402


COSMOS = "nvidia/Cosmos-Reason2-2B"


def local_snapshot(repo: str) -> str:
    """Local snapshot dir of a fully cached HF repo, else the repo id unchanged."""
    if pathlib.Path(repo).exists():
        return repo
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo, local_files_only=True)
    except Exception:  # not cached: let the stock loaders download it
        return repo


def vlm_files_from(path: str) -> None:
    """Load the Cosmos processor/config from `path` (a local snapshot) instead of the hub."""
    import gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 as processing

    real = processing.build_processor
    processing.build_processor = lambda name, kw: real(path if name == COSMOS else name, kw)


def vlm_from_config(vlm_files: str) -> None:
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    import gr00t.model.modules.qwen3_backbone as backbone

    config_kwargs = {"local_files_only", "trust_remote_code", "revision", "cache_dir", "token"}

    def from_pretrained(name, attn_implementation=None, torch_dtype=None, **kw):
        if set(kw) - config_kwargs:
            raise TypeError(f"unexpected VLM loading kwargs {sorted(set(kw) - config_kwargs)}")
        cfg = AutoConfig.from_pretrained(vlm_files if name == COSMOS else name, **kw)
        if attn_implementation:
            for c in (cfg, cfg.text_config, cfg.vision_config):
                c._attn_implementation = attn_implementation
        extra = {"torch_dtype": torch_dtype} if torch_dtype is not None else {}
        return Qwen3VLForConditionalGeneration._from_config(cfg, **extra)

    backbone.Qwen3VLForConditionalGeneration = type(
        "Qwen3VLFromConfig", (), {"from_pretrained": staticmethod(from_pretrained)})


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--rtc-max-delay", type=int, default=0)
    p.add_argument("--rtc-delay-dist", choices=("uniform", "exp"), default="uniform")
    p.add_argument("--vlm-weights", choices=("from-config", "download"), default="from-config")
    p.add_argument("--vlm-files", help=f"{COSMOS} config/processor dir (default: its cached snapshot)")
    own, rest = p.parse_known_args()
    if rest[:1] == ["--"]:
        rest = rest[1:]
    if "--modality-config-path" not in rest:
        rest += ["--modality-config-path", str(HERE / "modality.py")]
    if "--embodiment-tag" not in rest:
        rest += ["--embodiment-tag", "NEW_EMBODIMENT"]
    if "--output-dir" not in rest:
        sys.exit("--output-dir is required")
    out = pathlib.Path(rest[rest.index("--output-dir") + 1])

    if "--base-model-path" in rest:
        i = rest.index("--base-model-path") + 1
        rest[i] = local_snapshot(rest[i])
    vlm_files = own.vlm_files or local_snapshot(COSMOS)
    vlm_files_from(vlm_files)
    if own.vlm_weights == "from-config":
        vlm_from_config(vlm_files)
    if own.rtc_max_delay > 0:
        rtc = rtc_patch.RtcTraining(own.rtc_max_delay, own.rtc_delay_dist)
        rtc_patch.apply(rtc)
        out.mkdir(parents=True, exist_ok=True)
        (out / "rtc_training.json").write_text(rtc.to_json() + "\n")
        print(f"training-time RTC: {rtc}")

    sys.argv = [launch_finetune.__file__, *rest]
    runpy.run_path(launch_finetune.__file__, run_name="__main__")


if __name__ == "__main__":
    main()
