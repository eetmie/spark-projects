"""Turn a raw RGB frame + instruction (+ state) into the named tensors the
SmolVLA graph expects. This is the only model-specific glue that stays in
Python — the engine handles the VLM + denoising loop itself.

Logical input roles (names baked into the ONNX differ between exports — see
io_spec.py for how they get matched to the engine's actual tensor names):

    image        float32  [1, 3, H, W]
    img_mask     bool     [1]
    lang_tokens  int64    [1, L]
    lang_masks   bool     [1, L]
    state        float32  [1, S]
    noise        float32  [1, chunk, action_dim]
"""

from __future__ import annotations

import logging

import numpy as np

LOG = logging.getLogger("smolvla_runtime.preprocess")


def resize_rgb(image_rgb: np.ndarray, size: int) -> np.ndarray:
    """Resize to size x size. PIL if available (quality), else a numpy fallback."""
    h, w = image_rgb.shape[:2]
    if h == size and w == size:
        return image_rgb
    try:
        from PIL import Image
        return np.asarray(Image.fromarray(image_rgb).resize((size, size), Image.BILINEAR))
    except ImportError:
        ys = (np.arange(size) * (h / size)).astype(int)
        xs = (np.arange(size) * (w / size)).astype(int)
        return image_rgb[ys][:, xs]


class InputBuilder:
    """Builds the dict of named numpy tensors for one inference call.

    Dims come from the engine's I/O spec, so this stays consistent with whatever
    the Spark export baked in (image size, language length, state/action dims).
    """

    def __init__(
        self,
        model_id: str,
        image_size: int,
        lang_max_len: int,
        state_dim: int,
        chunk_size: int,
        action_dim: int,
        fixed_noise: bool = False,
        seed: int = 0,
    ):
        self.image_size = image_size
        self.lang_max_len = lang_max_len
        self.state_dim = state_dim
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self._rng = np.random.default_rng(seed)
        self._fixed_noise = fixed_noise
        self._cached_noise = (
            self._rng.standard_normal((1, chunk_size, action_dim)).astype(np.float32)
            if fixed_noise else None
        )
        self._tokenizer = self._load_tokenizer(model_id)

    @staticmethod
    def _load_tokenizer(model_id: str):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        LOG.info("Tokenizer loaded from %s", model_id)
        return tok

    def _noise(self) -> np.ndarray:
        # Flow-matching wants fresh noise per call; --fixed-noise reuses one draw
        # so benchmark latency isn't muddied by RNG and outputs are reproducible.
        if self._cached_noise is not None:
            return self._cached_noise
        return self._rng.standard_normal((1, self.chunk_size, self.action_dim)).astype(np.float32)

    def _image(self, image_rgb: np.ndarray) -> np.ndarray:
        arr = resize_rgb(image_rgb, self.image_size).astype(np.float32) / 255.0
        return arr.transpose(2, 0, 1)[np.newaxis]  # [1, 3, H, W]

    def _state(self, state: np.ndarray | None) -> np.ndarray:
        if state is None:
            raw = np.zeros(self.state_dim, dtype=np.float32)
        else:
            raw = np.asarray(state, dtype=np.float32).reshape(-1)
            if raw.size < self.state_dim:
                raw = np.concatenate([raw, np.zeros(self.state_dim - raw.size, dtype=np.float32)])
            else:
                raw = raw[: self.state_dim]
        return raw[np.newaxis]  # [1, S]

    def build(self, image_rgb: np.ndarray, instruction: str, state: np.ndarray | None) -> dict:
        enc = self._tokenizer(
            instruction or "",
            max_length=self.lang_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        return {
            "image": self._image(image_rgb),
            "img_mask": np.ones((1,), dtype=bool),
            "lang_tokens": enc["input_ids"].astype(np.int64),
            "lang_masks": enc["attention_mask"].astype(bool),
            "state": self._state(state),
            "noise": self._noise(),
        }
