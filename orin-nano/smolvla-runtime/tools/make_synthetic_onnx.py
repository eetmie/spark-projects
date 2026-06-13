#!/usr/bin/env python3
"""Emit a SHAPE-CORRECT but trivial SmolVLA ONNX for plumbing tests.

This is NOT the model. It has the exact input/output interface the runtime's
io_spec expects (image0/img_mask0/lang_tokens/lang_masks/state/noise -> actions),
so it exercises the *whole* runtime on-device — ORT session creation, the TensorRT
EP build + engine cache, io_spec resolution, preprocess (real tokenizer), and the
run_pipeline / parity loops — WITHOUT needing the real ONNX exported on the Spark.

The graph is `actions = noise` (+ a zero-weighted touch of every other input so
none get dead-stripped), so outputs are meaningless. Use it only to prove the JP7.2
runtime works end to end; swap in the real Spark ONNX for actual inference.

    python tools/make_synthetic_onnx.py --out exports/synthetic_smolvla.onnx
"""
from __future__ import annotations

import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build(image_size=512, lang_len=48, state_dim=32, chunk=50, action_dim=32):
    B = 1
    inputs = [
        helper.make_tensor_value_info("image0", TensorProto.FLOAT, [B, 3, image_size, image_size]),
        helper.make_tensor_value_info("img_mask0", TensorProto.BOOL, [B]),
        helper.make_tensor_value_info("lang_tokens", TensorProto.INT64, [B, lang_len]),
        helper.make_tensor_value_info("lang_masks", TensorProto.BOOL, [B, lang_len]),
        helper.make_tensor_value_info("state", TensorProto.FLOAT, [B, state_dim]),
        helper.make_tensor_value_info("noise", TensorProto.FLOAT, [B, chunk, action_dim]),
    ]
    outputs = [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [B, chunk, action_dim])]

    # actions = noise + 0*mean(state) + 0*mean(image)  — touches the big float inputs
    # so the graph keeps them live, but contributes nothing numerically. Use ReduceMean
    # (not Sum) so the aux term stays well within FP16 range (a Sum over 512x512 would
    # overflow FP16 -> inf -> nan once the TRT EP builds the engine in fp16).
    zero = numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="zero")
    nodes = [
        helper.make_node("ReduceMean", ["state"], ["state_mean"], keepdims=0),
        helper.make_node("ReduceMean", ["image0"], ["image_mean"], keepdims=0),
        helper.make_node("Add", ["state_mean", "image_mean"], ["aux_sum"]),
        helper.make_node("Mul", ["aux_sum", "zero"], ["aux_zero"]),
        helper.make_node("Add", ["noise", "aux_zero"], ["actions"]),
    ]
    graph = helper.make_graph(nodes, "synthetic_smolvla", inputs, outputs, initializer=[zero])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exports/synthetic_smolvla.onnx")
    args = ap.parse_args()
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    onnx.save(build(), args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
