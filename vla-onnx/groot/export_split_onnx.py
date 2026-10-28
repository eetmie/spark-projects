"""ONNX export helpers shared by export_split_onnx17.py: graph dump, mixed-FP16 conversion
(RMSNorm kept FP32 by name, FP16 round trips stripped) and duplicate-Cast cleanup.

The N1.6 exporter that also lived here was removed 2026-10-28.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import onnx
import torch

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent / "common", HERE.parent):    # spark-projects layout, repo export/ layout
    if (_p / "vla_common").is_dir():
        sys.path.insert(0, str(_p))

import re  # noqa: E402

from vla_common.bundle import write_manifest  # noqa: E402, F401  (re-exported to export_split_onnx17)
from vla_common.fp16_weights import FP16_SENSITIVE_OPS  # noqa: E402

from groot_split import n_params  # noqa: E402

SENSITIVE = tuple(FP16_SENSITIVE_OPS)

# Every node of Qwen3's RMSNorms stays FP32, matched by name, not op type. Layer 2 writes
# a ~16150 "massive activation" into the sink token's residual and every later layer
# carries it: mean(x^2) is then ~1.3e5, and the `+ eps` Add between ReduceMean and Sqrt
# overflows FP16 to inf, rsqrt -> 0, the token is zeroed. Blocking Pow/ReduceMean alone
# left that Add in FP16 (measured: llm_7 cos 0.567, actions cos 0.992). Stock PyTorch
# upcasts the whole norm, so this is what the reference does too.
RMSNORM_NODE = re.compile(r"(input_layernorm|post_attention_layernorm|q_norm|k_norm|^/norm)/")


def strip_fp16_roundtrips(m: onnx.ModelProto) -> int:
    """Remove Cast(FP32->FP16) -> Cast(->FP32) pairs; consumers read the FP32 source.

    ORT's float16 converter inserts this round trip on edges BETWEEN two blocked
    (FP32-kept) nodes, e.g. RMSNorm's Pow -> ReduceMean. With Qwen3's massive activation
    x^2 is ~2.6e8, the FP16 hop makes it inf, and the "FP32" norm returns 0. Dropping
    the pair is strictly more precise and changes no other node.
    """
    typed = onnx.shape_inference.infer_shapes(m)
    types = {v.name: v.type.tensor_type.elem_type
             for v in list(typed.graph.value_info) + list(typed.graph.input) + list(typed.graph.output)}
    types.update({i.name: i.data_type for i in m.graph.initializer})
    prod = {o: n for n in m.graph.node for o in n.output}
    outputs = {o.name for o in m.graph.output}

    def to(n):
        return next(a.i for a in n.attribute if a.name == "to")

    F32, F16 = onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16
    rename = {}
    for n in m.graph.node:
        if n.op_type == "Cast" and to(n) == F32 and n.output[0] not in outputs:
            p = prod.get(n.input[0])
            if p is not None and p.op_type == "Cast" and to(p) == F16 and types.get(p.input[0]) == F32:
                rename[n.output[0]] = p.input[0]
    for n in m.graph.node:
        for i, x in enumerate(n.input):
            if x in rename:
                n.input[i] = rename[x]
    while True:                      # drop nodes nothing reads any more
        used = {x for n in m.graph.node for x in n.input} | outputs
        keep = [n for n in m.graph.node if any(o in used for o in n.output)]
        if len(keep) == len(m.graph.node):
            break
        del m.graph.node[:]
        m.graph.node.extend(keep)
    return len(rename)


def to_mixed_fp16(src: Path, dst: Path) -> None:
    from onnxruntime.transformers import float16
    from onnxruntime.transformers.onnx_model import OnnxModel

    m = onnx.load(str(src))
    nodes = [n.name for n in m.graph.node if RMSNORM_NODE.search(n.name)]
    out = float16.convert_float_to_float16(
        m, keep_io_types=True, node_block_list=nodes,
        op_block_list=list(float16.DEFAULT_OP_BLOCK_LIST) + list(SENSITIVE))
    n = strip_fp16_roundtrips(out)
    if n:
        print(f"  {src.stem}: removed {n} FP32->FP16->FP32 round trips")
    # save_model_to_file topologically sorts (keep_io_types appends input Casts last).
    OnnxModel(out).save_model_to_file(str(dst), use_external_data_format=False)


def dump(module, args, path: Path, inputs, outputs):
    t0 = time.time()
    torch.onnx.export(module, args, str(path), input_names=inputs, output_names=outputs,
                      opset_version=18, dynamo=False, do_constant_folding=True)
    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    empty = [n.name for n in m.graph.node if any(i == "" for i in n.input)
             and n.op_type not in ("Resize", "Clip", "Pad", "Slice")]
    if empty:
        raise SystemExit(f"{path.name}: nodes with empty inputs {empty[:5]}")
    print(f"  {path.name:14s} {n_params(module) / 1e6:6.1f} M  "
          f"{path.stat().st_size / 1e6:6.0f} MB  {time.time() - t0:4.0f} s")


def dedupe_casts(path: Path) -> int:
    """ORT's float16 converter can insert the same `*_cast_to_fp32` Cast twice when one
    tensor feeds two FP32-kept ops (seen in dit_0's attention-mask path): same input,
    same output name, which onnx and ORT reject as non-SSA. Drop the exact repeats."""
    m = onnx.load(str(path))
    seen, keep = {}, []
    for n in m.graph.node:
        key = tuple(n.output)
        if key in seen:
            first = seen[key]
            assert (first.op_type, list(first.input), first.attribute) == \
                (n.op_type, list(n.input), n.attribute), f"{path.name}: {key} defined twice, differently"
            continue
        seen[key] = n
        keep.append(n)
    dropped = len(m.graph.node) - len(keep)
    if dropped:
        del m.graph.node[:]
        m.graph.node.extend(keep)
        onnx.save(m, str(path))
        onnx.checker.check_model(str(path))
    return dropped


