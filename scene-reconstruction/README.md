# scene-reconstruction

Just use Spirula Studio (not sponsored haha). On the Spark (GB10), add `vkQueueWaitIdle()`
after the normal timeline wait in `spirula-studio/src/nn/vk/Stream.cpp` (enabled by
`SS_NN_QUEUE_IDLE_SYNC=1`), otherwise depth generation freezes.

What is left here: Gaussian PLY -> NuRec USDZ for Isaac Sim.

Needs a Python with `pxr` + `msgpack`: Isaac Lab's venv has `pxr` (`msgpack` installed there 2026-10-28).

```bash
export USD_CONVERT_PYTHON=~/IsaacLab-3.0-ga/.venv/bin/python
python usd_convert.py scene.ply scene.usdz --align    # --align: Z-up, origin from the COLMAP cameras in sparse/0/
python usd_convert.py --help
```
