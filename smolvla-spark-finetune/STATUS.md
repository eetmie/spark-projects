# Status

Experimental but locally verified on NVIDIA GB10.

## Verified

- PyTorch CUDA works on NVIDIA GB10 with `torch==2.12.0+cu130`.
- LeRobot `0.5.1` imports and runs SmolVLA.
- SmolVLA CUDA forward works on GB10.
- Official `lerobot/svla_so101_pickplace` dataset was downloaded as a smoke-test fixture only.
- AV1 videos were transcoded to a local H.264 working copy for reliable LeRobot loading.
- A 1-step SmolVLA LoRA fine-tune smoke test completed on GB10 using the fixture dataset.
- Baseline SmolVLA ONNX export completed.
- ONNX checker passed.
- CPU ONNX Runtime session creation passed.
- PyTorch-vs-ONNX parity passed with `max_abs_diff ~= 2.62e-6` and cosine effectively `1.0`.

## Settled on-Orin (2026-06-17)

- **The monolithic ONNX does NOT TRT-build on the Orin Nano's 8 GB** — not FP32, not FP16, not
  `--num-steps 5`, not headless. TRT imports all 450M weights as FP32 working copies at once (~6 GB
  floor, independent of node count), so the build OOMs/thrashes. `--fp16-weights` and fewer steps do
  not fix it. (Full matrix: `orin-nano/smolvla-runtime/notes/findings.md`.)
- **The deploy path is SPLIT per-component engines** (vision / text / expert-prefill / expert-decode
  + projectors), denoise loop run in Python. Validated on-device with the reference base-weight split
  (`ainekko/smolvla_base_onnx`): each heavy engine builds in ≤60 s and runs in ms → ~5–9 Hz end-to-end.

## Verified 2026-08-12 (excavator fine-tune — see excavator/ and Desktop notes)

- Full fine-tune on a real dataset (`masi_kaivuri_juusto`, 31 eps): 4-run sweep completed,
  best held-out checkpoint run A step 6000 (disp_err 0.1349 vs 0.3901 zero-action baseline).
- **Split export of OUR fine-tuned weights**: `export_split_onnx.py --model-id <checkpoint>`
  → `exports-split-excavA6000/`. Split-vs-monolith parity on identical seeded inputs:
  cosine 1.0000000, max_abs 4.05e-6 → **PASS, Orin-ready**.
- Deploy bundle shipped to the Orin (`orin-nano/.../exports/excav_A6000_split/`): 9 graphs +
  tokenizer + stats.json (verified identical to the checkpoint's normalizer) +
  `export_info.json` with `fps` (read by kaivuriprokkis' run_inference fps guard) + PARITY.txt.
- Synthetic split-TRT inference on the Orin with base weights: engines build ≤60 s each,
  206–227 ms per inference.


## Verified 2026-08-13 (long sweep, A20k deploy, first successful live run)

- 30k-step sweep scored across all checkpoints: best = run A step 20000, disp_err 0.1320
  (vs 0.1413 at 8k); all four configs still tied within 2%; no real overfit by 30k.
- `exports-split-excavA20k` byte-verified against the A@020000 checkpoint (state_projector
  allclose vs safetensors) and deployed to the Orin as `excav_A20k_split` (+stats.json +
  export_info.json). Synthetic TRT proof: 209-217 ms/inference.
- Live valve pipeline instrumented on-device (probe monkeypatches update_named/reset):
  100 Hz writes sustained during real inference (worst 0.5 s window 86 Hz vs 50 Hz gate),
  no watchdog starvation. Machine standing still traced to the POLICY going quiet at the
  parked pose (bucket fully curled): 2x2 obs-swap showed dataset-state -> motion-scale
  actions even on the live image. Fix: start from the episode-start pose.
- **First successful live VLA run on the real excavator (2026-08-13).**
- Run E launched: state-blind ablation (observation.state zeroed in a dataset variant +
  stats patched to mean 0/std 1), otherwise identical to run A. Tests camera-only control.

## Verified 2026-08-14 (camera-only run E scored, exported, deployed)

- **Camera-only control works.** Run E finished 30k steps and was scored across all 12
  checkpoints: best = **E step 17500, disp_err 0.1387** (zero-action baseline 0.3901).
  That is only **+5.1% vs the state-fed A@020000 (0.1320)** — dropping the IMU costs
  almost nothing. Curve flat 0.1387-0.1422 from step 15000; no overfit by 30k (0.1410).
  Run A re-scored with the patched eval reproduced 0.1320 exactly (no regression).
- `eval_compare.py` gained `FPS_BY_REPO` + `STATE_BLIND_REPOS`: E's repo_id was a KeyError
  in the old hardcoded fps lookup, and eval must feed **zeroed** state for state-blind runs
  (the source dataset's real state reaches |120|, which E never saw).
- **The state input is provably dead, not removed.** `model.state_proj.weight` is
  bit-identical across ALL E checkpoints (max delta 0.0) while its bias drifts: a
  constant-zero input gives the weight zero gradient. So the weight still sits at its
  *pretrained* value and is nonzero — feeding real state does NOT get ignored, it injects
  a garbage prefix token. Measured: plausible angles instead of zeros shift the commanded
  action by up to **0.434 on the [-1,1] joystick scale** (mean 0.093).
- `exports-split-excavE17500` exported and provenance-checked: action_out_proj,
  action_in_proj and action_time_mlp_in all match E@017500 exactly and differ from
  E@015000/020000/030000 and from A@020000. Parity vs the **PyTorch checkpoint** (no
  monolith needed): cosine 1.0000000, max_abs 2.03e-6 at zero state, 2.38e-6 at random
  state. Shipped stats.json verified identical to the checkpoint's baked-in normalizer.
- Deployed to the Orin as `excav_E17500_split` (24/24 sha256 verified). On-device synthetic
  TRT proof: engines build 26-27 s each, **204-251 ms/inference**.
- `kaivuriprokkis/lerobot_vla/run_inference.py` gained `resolve_state_blind()`: it reads
  `state_blind` from export_info.json and feeds the policy zeros (the IMU value is still
  logged, marked `state(unused)=`). Additive and flag-gated — A20k re-verified unchanged
  (no warning, `state=`, 215 ms). Without this the bundle loads and runs *fine* and drives
  the machine wrong, because the state tensor is still in the interface.

## Not Yet Verified

- Exporting a fine-tuned LoRA checkpoint after a real training run (excavator runs are full
  expert fine-tunes, not LoRA).
- Merging LoRA weights into a self-contained checkpoint for deployment export.
- On-Orin FP16-vs-FP32 parity of the split pipeline.
- Real robot preprocessing/postprocessing loop end-to-end (`--live`).
- Real SO-101 / SO-100 hardware inference.

## Current Intended Workflow

1. Fine-tune SmolVLA on GB10 using LeRobot.
2. Export the **split** graphs on GB10 (vision, text, expert-prefill[KV out], expert-decode[KV in,
   single step], + the projectors) — NOT the monolithic `sample_actions`. Blueprint: `ainekko/
   smolvla_base_onnx` + `github.com/aifoundry-org/ETARS` (`smolVLA_export.ipynb`,
   `smolvlm_with_expert_onnx.py`). The monolithic export (`export_valid_onnx.py`) stays only as the
   FP32 parity gold on a big box.
3. Run PyTorch-vs-ONNX parity on GB10.
4. Copy the split bundle (9 graphs + `tokenizer/` + normalization stats) to the Orin Nano.
5. The Orin builds + caches one TRT engine per heavy graph (FP16, ≤60 s each), and runs the
   prefill→decode loop in Python (`orin-nano/smolvla-runtime/backends/ort.py`).
6. Run on-Orin parity, then robot integration.
