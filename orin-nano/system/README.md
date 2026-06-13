# system — Orin Nano Super performance + memory setup (JetPack 7.2)

One-time host setup the SmolVLA runtime depends on.

## 1. Max performance: MAXN_SUPER + pinned clocks

```bash
./power-max.sh                    # nvpmodel -m 2 (MAXN_SUPER) + jetson_clocks, now
```

Make it persistent across reboots:

```bash
sudo cp jetson-perf.service /etc/systemd/system/jetson-perf.service
sudo systemctl daemon-reload
sudo systemctl enable --now jetson-perf.service
```

nvpmodel IDs on this board (confirm with `sudo nvpmodel -p --verbose`):
`0 = 15W`, `1 = 25W`, `2 = MAXN_SUPER`.

## 2. Swap for the TensorRT engine build

The Orin Nano has **8 GB unified** memory (CPU + GPU share it). The first ORT
TensorRT-EP engine build is the memory peak and OOMs on the stock 2 GB swap. Grow it:

```bash
sudo ./setup-swap.sh              # 16 GB swapfile on the NVMe + vm.swappiness=10, persisted
```

Before a big build, also stop idle GPU/memory hogs — the memory is shared, so a
background server eats into what TensorRT can use.

## 3. onnxruntime-gpu for CUDA 13

The runtime needs `onnxruntime-gpu` built for **CUDA 13 + cp312 + aarch64**. There is
no PyPI wheel, and there is **no `jp7` index** on Jetson AI Lab — the CUDA-13 aarch64
builds live under the **`sbsa`** index. Verified working: `onnxruntime-gpu==1.24.0`.

```bash
pip install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/sbsa/cu130
python -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())"
# -> 1.24.0  ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

A benign warning `GPU device discovery failed: .../card1/device/vendor` may print —
it's a Jetson iGPU/DRM-probe quirk; the CUDA EP still works via CUDA. The same index
also has `cuda-python 13`, `torch 2.11`, etc. if ever needed.
