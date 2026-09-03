from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch

SCRIPT = Path(__file__).parents[1] / "tools" / "convert_mint_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("convert_mint_checkpoint", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ConvertMintCheckpointTest(unittest.TestCase):
    def test_action_and_language_keys(self) -> None:
        source = {
            "action_head.norm_out.weight": torch.ones(3),
            "embedder.model.language_model.model.norm.weight": torch.ones(4),
        }
        converted = MODULE.convert_module_state(source)
        self.assertEqual(
            set(converted),
            {
                "model.action_head.norm_out.weight",
                "model.embedder.model.language_model.norm.weight",
            },
        )

    def test_vision_qkv_is_split_without_changing_elements(self) -> None:
        packed = torch.arange(18).reshape(6, 3)
        source = {
            "embedder.model.vision_model.encoder.layers.2.attn.qkv.weight": packed
        }
        converted = MODULE.convert_module_state(source, policy_prefix=False)
        names = [
            f"embedder.model.vision_tower.encoder.layer.2.attention.{item}.weight"
            for item in ("q_proj", "k_proj", "v_proj")
        ]
        self.assertEqual(set(converted), set(names))
        self.assertTrue(torch.equal(converted[names[0]], packed[:2]))
        self.assertTrue(torch.equal(converted[names[1]], packed[2:4]))
        self.assertTrue(torch.equal(converted[names[2]], packed[4:]))

    def test_unknown_keys_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported MINT EVO1 key"):
            MODULE.convert_module_state({"unknown.weight": torch.ones(1)})


if __name__ == "__main__":
    unittest.main()
