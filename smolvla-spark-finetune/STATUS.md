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

## Not Yet Verified

- Exporting a fine-tuned LoRA checkpoint after a real training run.
- Merging LoRA weights into a self-contained checkpoint for deployment export.
- **Re-exporting OUR fine-tuned weights in the SPLIT layout** (next big task — see NOTES.md).
- On-Orin FP16-vs-FP32 parity of the split pipeline.
- Real robot preprocessing/postprocessing loop.
- Useful task-specific behavior from the fixture dataset.
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
