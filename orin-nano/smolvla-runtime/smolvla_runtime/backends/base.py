"""Backend interface for the SmolVLA Orin-Nano runtime.

A backend takes one RGB frame + a language instruction (+ optional robot state)
and returns an action chunk. The pipeline does not care *how* the actions are
produced — the ORT TensorRT-EP runtime or a mock — only that every backend
honours this contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class PredictResult:
    actions: np.ndarray          # [chunk_size, action_dim], float32
    latency_ms: float            # wall-clock for the inference call only
    backend: str                 # short backend id, e.g. "ort"
    extra: dict = field(default_factory=dict)


class Backend(Protocol):
    """Anything the pipeline can drive."""

    description: str

    def predict(
        self,
        image_rgb: np.ndarray,
        instruction: str,
        state: np.ndarray | None,
    ) -> PredictResult:
        ...
