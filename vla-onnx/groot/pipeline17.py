"""Host-side glue for a GR00T N1.7 split bundle. numpy only; the graph runner is injected.

`run(name, feeds) -> dict` is the only thing a backend provides: ORT sessions on the
Spark for parity, TensorRT engines on the Orin.
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
        self.graph = {g["name"]: g for g in self.b["graphs"]}
        self.vision = [n for n in self.names if n.startswith("vision_")]
        self.llm = [n for n in self.names if n.startswith("llm_")]
        self.cond = [n for n in self.names if n.startswith("cond_")]
        self.dit = [n for n in self.names if n.startswith("dit_")]
        ids = np.asarray(self.b["input_ids"], dtype=np.int64)
        self.ids = ids
        self.img = ids == self.b["image_token"]
        n, neg = self.b["prompt_tokens"], np.float32(self.b["mask_neg"])
        valid = np.arange(ids.shape[0]) < n
        self.text_bias = np.where(~self.img & valid, 0.0, neg).astype(np.float32)[None, None]
        self.image_bias = np.where(self.img & valid, 0.0, neg).astype(np.float32)[None, None]
        self.pad_bias = np.where(valid, 0.0, neg).astype(np.float32)[None, None]
        # The prompt never changes within a bundle: its embedded rows are gathered once.
        self.prompt_embeds = np.asarray(self.embed[ids], dtype=np.float32)[None]

    def normalize(self, images_u8: np.ndarray) -> np.ndarray:
        """[N,H,W,3] uint8, already at image_hw -> [N,3,H,W] float32."""
        x = images_u8.astype(np.float32) / 255.0
        x = (x - self.b["image_mean"]) / self.b["image_std"]
        return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def encode_frames(bundle: Bundle, run, pixel_values: np.ndarray) -> list[np.ndarray]:
    """One frame per view [V,3,H,W] -> [tokens, ds_0, ds_1, ...], each [V,T,2048]."""
    x, ds = pixel_values, {}
    for name in bundle.vision:
        o = run(name, {"pixel_values" if name == bundle.vision[0] else "x": x})
        x = o["vision_tokens" if name == bundle.vision[-1] else "x_out"]
        ds.update({k: v for k, v in o.items() if k.startswith("ds_")})
    return [x, *[ds[f"ds_{j}"] for j in range(bundle.b["deepstack"])]]


def infer(bundle: Bundle, run, frames: list[list[np.ndarray]], state: np.ndarray,
          noise: np.ndarray, timings: dict | None = None) -> np.ndarray:
    """One action chunk.

    frames: per time slot, oldest first, the encode_frames() output of that slot's views.
    state [1,1,132] normalized, noise [1,40,132].
    """
    b = bundle.b
    t = {} if timings is None else timings
    t0 = time.perf_counter()
    h = bundle.prompt_embeds.copy()
    sel = bundle.img
    h[0, sel] = np.concatenate([f[0].reshape(-1, f[0].shape[-1]) for f in frames])
    ds_full = []
    for j in range(b["deepstack"]):
        z = np.zeros_like(h)
        z[0, sel] = np.concatenate([f[1 + j].reshape(-1, f[1 + j].shape[-1]) for f in frames])
        ds_full.append(z)
    for name in bundle.llm:
        feeds = {"h": h}
        feeds.update({k: ds_full[int(k[3:])] for k in bundle.graph[name]["inputs"] if k.startswith("ds_")})
        h = next(iter(run(name, feeds).values()))
    o = run(bundle.cond[0], {"features": h, "pad_bias": bundle.pad_bias,
                             "state": state.astype(np.float32)})
    vl, sf = o["vl"], o["state_features"]
    for name in bundle.cond[1:]:
        vl = run(name, {"vl": vl, "pad_bias": bundle.pad_bias})["vl_out"]
    tb, ib = bundle.text_bias, bundle.image_bias
    t1 = time.perf_counter()

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
    t2 = time.perf_counter()
    t.update(backbone_ms=(t1 - t0) * 1e3, dit_ms=(t2 - t1) * 1e3)
    return actions
