import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from xvla_runtime.bundle_contract import (
    SCHEMA_VERSION,
    build_processor_contract,
    copy_processor_artifacts,
    normalize_vector,
    tree_sha256,
    unnormalize_vector,
    verify_bundle,
)
from xvla_runtime.split_ort import (
    XVLASplitPolicy,
    _validate_engine_cache_manifest,
    _write_engine_cache_manifest,
)
from run_pipeline import validate_runtime_args


class BundleContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.checkpoint = Path(self.temp.name) / "checkpoint"
        self.checkpoint.mkdir()
        config = {
            "input_features": {
                "observation.state": {"type": "STATE", "shape": [3]},
                "observation.images.cam1": {"type": "VISUAL", "shape": [3, 8, 8]},
            },
            "output_features": {"action": {"type": "ACTION", "shape": [4]}},
            "max_state_dim": 20,
            "max_action_dim": 20,
            "action_mode": "auto",
        }
        pre = {
            "steps": [
                {
                    "registry_name": "tokenizer_processor",
                    "config": {
                        "max_length": 50,
                        "tokenizer_name": "facebook/bart-large",
                        "padding": "max_length",
                        "padding_side": "right",
                        "truncation": True,
                    },
                },
                {
                    "registry_name": "normalizer_processor",
                    "config": {"eps": 1e-8, "norm_map": {"STATE": "MEAN_STD"}},
                    "state_file": "pre.safetensors",
                },
            ]
        }
        post = {
            "steps": [
                {
                    "registry_name": "unnormalizer_processor",
                    "config": {"eps": 1e-8, "norm_map": {"ACTION": "MEAN_STD"}},
                    "state_file": "post.safetensors",
                }
            ]
        }
        (self.checkpoint / "config.json").write_text(json.dumps(config))
        (self.checkpoint / "policy_preprocessor.json").write_text(json.dumps(pre))
        (self.checkpoint / "policy_postprocessor.json").write_text(json.dumps(post))
        save_file(
            {
                "observation.state.mean": np.array([1, 2, 3], dtype=np.float32),
                "observation.state.std": np.array([2, 4, 8], dtype=np.float32),
            },
            self.checkpoint / "pre.safetensors",
        )
        save_file(
            {
                "action.mean": np.array([10, 20, 30, 40], dtype=np.float32),
                "action.std": np.array([1, 2, 3, 4], dtype=np.float32),
            },
            self.checkpoint / "post.safetensors",
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_extracts_and_applies_physical_boundary(self):
        contract = build_processor_contract(self.checkpoint)
        self.assertTrue(contract["physical_boundary_complete"])
        self.assertEqual(contract["state"]["dim"], 3)
        self.assertEqual(contract["state"]["model_dim"], 20)
        self.assertEqual(contract["action"]["dim"], 4)
        self.assertEqual(contract["action"]["model_dim"], 20)
        self.assertEqual(contract["tokenizer"]["max_length"], 50)

        normalized = normalize_vector(
            np.array([3, 6, 11], dtype=np.float32), contract["state"])
        np.testing.assert_allclose(normalized, [1, 1, 1])
        physical = unnormalize_vector(
            np.ones((2, 4), dtype=np.float32), contract["action"])
        np.testing.assert_allclose(physical[0], [11, 22, 33, 44])

    def test_bundle_verification_fails_on_tokenizer_tamper(self):
        contract = build_processor_contract(self.checkpoint)
        bundle_dir = Path(self.temp.name) / "bundle"
        bundle_dir.mkdir()
        copy_processor_artifacts(self.checkpoint, bundle_dir, contract)
        tokenizer_dir = bundle_dir / "tokenizer"
        tokenizer_dir.mkdir()
        (tokenizer_dir / "tokenizer.json").write_text("{}")
        bundle = {
            "schema_version": SCHEMA_VERSION,
            "checkpoint": {"tree_sha256": tree_sha256(self.checkpoint)},
            "processor_contract": contract,
            "tokenizer": {
                "path": "tokenizer",
                "tree_sha256": tree_sha256(tokenizer_dir),
            },
            "graphs": [],
        }
        (bundle_dir / "bundle.json").write_text(json.dumps(bundle))
        self.assertEqual(verify_bundle(bundle_dir), bundle)

        (tokenizer_dir / "tokenizer.json").write_text('{"tampered": true}')
        with self.assertRaisesRegex(ValueError, "tokenizer tree"):
            verify_bundle(bundle_dir)

    def test_runtime_crosses_the_physical_boundary_once(self):
        contract = build_processor_contract(self.checkpoint)
        policy = object.__new__(XVLASplitPolicy)
        policy.state_dim = 3
        policy.model_state_dim = 20
        policy.real_action_dim = 4
        policy.action_dim = 20
        policy.chunk_size = 2
        policy.steps = 1
        policy.denoise_input_mode = "x_t"
        policy.gripper_idx = ()
        policy.processor_contract = contract
        policy.rng = np.random.default_rng(0)
        policy.last_timings = {}
        policy.encode_observation = lambda images, instruction: np.zeros((1, 1, 1))
        captured = {}

        def denoise(x_t, t, proprio, cond_tokens):
            captured["proprio"] = proprio.copy()
            return np.ones_like(x_t)

        policy._denoise_step = denoise
        actions = policy.sample_actions(
            [], "test", np.array([3, 6, 11], dtype=np.float32),
            x1=np.zeros((1, 2, 20), dtype=np.float32),
        )
        np.testing.assert_allclose(captured["proprio"][0, :3], [1, 1, 1])
        np.testing.assert_allclose(captured["proprio"][0, 3:], 0)
        np.testing.assert_allclose(policy.last_normalized_action, 1)
        np.testing.assert_allclose(actions[0], [11, 22, 33, 44])
        self.assertEqual(actions.shape, (2, 4))

    def test_missing_declared_stats_is_identity_but_not_deployable_auto(self):
        pre_path = self.checkpoint / "policy_preprocessor.json"
        post_path = self.checkpoint / "policy_postprocessor.json"
        pre = json.loads(pre_path.read_text())
        post = json.loads(post_path.read_text())
        pre["steps"][1].pop("state_file")
        post["steps"][0].pop("state_file")
        pre_path.write_text(json.dumps(pre))
        post_path.write_text(json.dumps(post))

        contract = build_processor_contract(self.checkpoint)
        self.assertFalse(contract["physical_boundary_complete"])
        self.assertEqual(contract["state"]["normalization"]["mode"], "IDENTITY")
        self.assertEqual(
            contract["state"]["normalization"]["declared_mode"], "MEAN_STD")

    def test_device_resident_chain_binds_hidden_outputs_on_cuda(self):
        class Value:
            def __init__(self, array):
                self.array = np.asarray(array)

            def numpy(self):
                return self.array

        class Output:
            def __init__(self, name):
                self.name = name

        class Binding:
            def __init__(self):
                self.inputs = {}
                self.outputs = []
                self.output_devices = []

            def clear_binding_inputs(self):
                self.inputs.clear()

            def clear_binding_outputs(self):
                self.outputs.clear()

            def bind_cpu_input(self, name, value):
                self.inputs[name] = Value(value)

            def bind_ortvalue_input(self, name, value):
                self.inputs[name] = value

            def bind_output(self, name, device, device_id):
                self.output_devices.append((name, device, device_id))

            def get_outputs(self):
                return self.outputs

        class Session:
            def __init__(self, index):
                self.index = index
                self.binding = Binding()

            def get_outputs(self):
                return [Output("action" if self.index == 3 else "hidden_out")]

            def run_with_iobinding(self, binding):
                source = (binding.inputs.get("x_t") or binding.inputs.get("action")
                          or binding.inputs["hidden_in"])
                binding.outputs = [Value(source.numpy() + 1)]

        policy = object.__new__(XVLASplitPolicy)
        policy.denoise = [Session(i) for i in range(4)]
        policy._denoise_io = [session.binding for session in policy.denoise]
        static = {
            "proprio": Value(np.zeros((1, 20), dtype=np.float32)),
            "cond_tokens": Value(np.zeros((1, 2, 3), dtype=np.float32)),
        }

        output = policy._denoise_step_device_resident(
            np.zeros((1, 2, 20), dtype=np.float32), 1.0, static)

        np.testing.assert_allclose(output, 4)
        self.assertIs(policy._denoise_io[0].inputs["proprio"], static["proprio"])
        self.assertIs(policy._denoise_io[0].inputs["cond_tokens"],
                      static["cond_tokens"])
        for binding in policy._denoise_io:
            self.assertEqual(binding.output_devices[-1][1:], ("cuda", 0))

        x1_device = Value(np.ones((1, 2, 20), dtype=np.float32))
        action_device = Value(np.zeros((1, 2, 20), dtype=np.float32))
        fused = policy._denoise_step_device_resident_fused(
            x1_device, action_device, 1.0, static)
        self.assertIs(policy._denoise_io[0].inputs["x1"], x1_device)
        self.assertIs(policy._denoise_io[0].inputs["action"], action_device)
        self.assertNotIn("x_t", policy._denoise_io[0].inputs)
        np.testing.assert_allclose(fused.numpy(), 4)

    def test_engine_cache_manifest_rejects_stale_or_mixed_contents(self):
        cache = Path(self.temp.name) / "cache"
        cache.mkdir()
        engine = cache / "graph.engine"
        engine.write_bytes(b"engine")
        identity = {"bundle": "abc", "precision": "fp16"}

        _write_engine_cache_manifest(cache, identity)
        document = _validate_engine_cache_manifest(cache, identity)
        self.assertEqual(document["identity"], identity)

        (cache / "mixed.timing").write_bytes(b"timing")
        with self.assertRaisesRegex(ValueError, "missing, truncated, or mixed"):
            _validate_engine_cache_manifest(cache, identity)
        (cache / "mixed.timing").unlink()
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            _validate_engine_cache_manifest(
                cache, {"bundle": "different", "precision": "fp16"})

    def test_direct_runner_rejects_non_positive_runtime_arguments(self):
        for duration, steps, report_every in (
                (0, None, 0), (-1, None, 0), (1, 0, 0), (1, -1, 0),
                (1, None, -1)):
            with self.subTest(
                    duration=duration, steps=steps, report_every=report_every):
                with self.assertRaises(ValueError):
                    validate_runtime_args(duration, steps, report_every)
        validate_runtime_args(1, None, 0)
        validate_runtime_args(1, 10, 0.5)

    def test_profile_sessions_flush_once(self):
        class Session:
            def __init__(self, path):
                self.path = path
                self.calls = 0

            def end_profiling(self):
                self.calls += 1
                return self.path

        policy = object.__new__(XVLASplitPolicy)
        first, second = Session("/tmp/vision.json"), Session("/tmp/denoise.json")
        policy._profile_sessions = {"vision_0": first, "denoise_0": second}
        self.assertEqual(policy.end_profiling(), {
            "vision_0": "/tmp/vision.json",
            "denoise_0": "/tmp/denoise.json",
        })
        self.assertEqual(policy.end_profiling(), {})
        self.assertEqual((first.calls, second.calls), (1, 1))


if __name__ == "__main__":
    unittest.main()
