from __future__ import annotations

import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "inspect_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("inspect_checkpoint", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class InspectCheckpointTest(unittest.TestCase):
    def test_reads_header_only_and_normalizes_lerobot_names(self) -> None:
        header = {
            "model.embedder.model.vision_tower.encoder.layer.0.weight": {
                "dtype": "BF16",
                "shape": [3, 4],
                "data_offsets": [0, 24],
            },
            "model.action_head.transformer_blocks.0.attn.in_proj_weight": {
                "dtype": "F32",
                "shape": [12, 4],
                "data_offsets": [24, 216],
            },
        }
        encoded = json.dumps(header).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "header-only.safetensors"
            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)
            tensors = MODULE.read_header(path)

        groups = MODULE.grouped_elements(tensors)
        self.assertEqual(groups["vision.layer.00"], 12)
        self.assertEqual(groups["action.block.00"], 48)
        self.assertEqual(MODULE.action_kv_elements(tensors), 32)

    def test_profile_keeps_oversized_token_table_off_trt(self) -> None:
        groups = {
            "language.token_embedding": 135_000_000,
            "language.layer.00": 15_000_000,
            "language.norm": 1_000,
        }
        profile = MODULE.split_profile([], groups, 100_000_000)
        self.assertIn(("ORT CPU", "language.token_embedding", 135_000_000), profile)
        self.assertFalse(
            any(
                backend.startswith("TRT") and count > 100_000_000
                for backend, _, count in profile
            )
        )


if __name__ == "__main__":
    unittest.main()
