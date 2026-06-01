"""Zero-action backend: exercises camera + loop plumbing with no model/engine.

Use this to confirm the RealSense feed and the reporting loop work before any
ONNX has been exported on the Spark.
"""

from __future__ import annotations

import numpy as np

from .base import PredictResult


class MockBackend:
    def __init__(self, chunk_size: int = 50, action_dim: int = 32):
        self._actions = np.zeros((chunk_size, action_dim), dtype=np.float32)
        self.description = f"mock zero-action chunk={chunk_size} dim={action_dim}"

    def predict(self, image_rgb, instruction, state) -> PredictResult:
        del image_rgb, instruction, state
        return PredictResult(actions=self._actions, latency_ms=0.0, backend="mock")
