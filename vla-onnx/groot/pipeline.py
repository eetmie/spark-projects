"""Host-side glue for a GR00T split bundle. numpy only; the graph runner is injected.

`run(name, feeds) -> dict` is the only thing a backend provides: ORT sessions on the
Spark for parity, TensorRT engines on the Orin. Everything else (embedding gather, image
token scatter, cross-attention biases, the denoising loop) lives here once.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


class Bundle:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.b = json.loads((self.root / "bundle.json").read_text())
        self.embed = np.load(self.root / self.b["embed_tokens"], mmap_mode="r")
        self.names = [g["name"] for g in self.b["graphs"]]
        self.vision = [n for n in self.names if n.startswith("vision_")]
        self.llm = [n for n in self.names if n.startswith("llm_")]
        self.dit = [n for n in self.names if n.startswith("dit_")]

    def biases(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, neg = self.b["prompt_tokens"], np.float32(self.b["mask_neg"])
        valid = np.arange(ids.shape[0]) < n
        img = ids == self.b["image_token"]
        tb = np.where(~img & valid, 0.0, neg).astype(np.float32)[None, None]
        ib = np.where(img & valid, 0.0, neg).astype(np.float32)[None, None]
        return tb, ib

    def normalize(self, images_u8: np.ndarray) -> np.ndarray:
        """[V,H,W,3] uint8, already at image_hw -> [V,3,H,W] float32."""
        x = images_u8.astype(np.float32) / 255.0
        x = (x - self.b["image_mean"]) / self.b["image_std"]
        return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def infer(bundle: Bundle, run, pixel_values: np.ndarray, state: np.ndarray, noise: np.ndarray,
          input_ids: np.ndarray | None = None, timings: dict | None = None) -> np.ndarray:
    """One action chunk. pixel_values [V,3,H,W] f32, state [1,1,128], noise [1,50,128]."""
    b = bundle.b
    ids = np.asarray(input_ids if input_ids is not None else b["input_ids"], dtype=np.int64)
    t = {} if timings is None else timings
    t0 = time.perf_counter()

    x = pixel_values
    for name in bundle.vision:
        x = next(iter(run(name, {"pixel_values" if name == bundle.vision[0] else "x": x}).values()))
    t1 = time.perf_counter()

    h = np.asarray(bundle.embed[ids], dtype=np.float32)[None]
    h[0, ids == b["image_token"]] = x.reshape(-1, x.shape[-1])
    for name in bundle.llm:
        h = next(iter(run(name, {"h": h}).values()))
    cond = run("cond", {"features": h, "state": state.astype(np.float32)})
    vl, sf = cond["vl"], cond["state_features"]
    tb, ib = bundle.biases(ids)
    t2 = time.perf_counter()

    actions = noise.astype(np.float32)
    for step in b["timesteps"]:
        te = run("time", {"t": np.array([step], np.float32)})
        o = run(bundle.dit[0], {"actions": actions, "t_proj": te["t_proj"], "tau": te["tau"],
                                "state_features": sf, "vl": vl, "text_bias": tb, "image_bias": ib})
        hh, temb = o["h_out"], o["temb"]
        for name in bundle.dit[1:-1]:
            hh = run(name, {"h": hh, "temb": temb, "vl": vl, "text_bias": tb, "image_bias": ib})["h_out"]
        actions = run(bundle.dit[-1], {"h": hh, "temb": temb, "vl": vl, "text_bias": tb,
                                       "image_bias": ib, "actions": actions})["actions_next"]
    t3 = time.perf_counter()
    t.update(vision_ms=(t1 - t0) * 1e3, backbone_ms=(t2 - t1) * 1e3, dit_ms=(t3 - t2) * 1e3,
             total_ms=(t3 - t0) * 1e3)
    return actions
