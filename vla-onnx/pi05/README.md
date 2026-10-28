# vla-onnx/pi05 — pi0.5 (OpenPI) excavator fine-tune and Orin export

```bash
./setup.sh                      # OpenPI c23745b in third_party/openpi + its .venv (GB10 pins inside)
```

- Fine-tune (configs, norm stats, training-time RTC): [../kaivuriprokkis/RTC_TRAINING.md](../kaivuriprokkis/RTC_TRAINING.md)
- Export to the Orin: `export/export.sh`; bundle layout in [BUNDLE.md](BUNDLE.md); robot-side exporter in [../kaivuriprokkis](../kaivuriprokkis/README.md)

Not in git: `third_party/`, `datasets/`, `checkpoints/` (local disk), `assets/` (norm stats), `runs/`.
