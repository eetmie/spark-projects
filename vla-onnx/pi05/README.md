# vla-onnx/pi05 — pi0.5 (OpenPI) excavator fine-tune and Orin export

```bash
./setup.sh                      # OpenPI c23745b in third_party/openpi + its .venv (GB10 pins inside)
```

- Fine-tune (configs, norm stats, training-time RTC): [../kaivuriprokkis/RTC_TRAINING.md](../kaivuriprokkis/RTC_TRAINING.md)
- Current run (2026-10-30): `digging_complete_train`, 10k steps, queued by `runs/queue_complete.sh`. Its audit and the demo plan are in RTC_TRAINING.md, "pi0.5 on digging_complete" and "Audit before the run".
- Export to the Orin: `export/export.sh`; bundle layout in [BUNDLE.md](BUNDLE.md); robot-side exporter in [../kaivuriprokkis](../kaivuriprokkis/README.md)

Not in git: `third_party/`, `datasets/`, `checkpoints/` (local disk), `assets/` (norm stats), `runs/`.
