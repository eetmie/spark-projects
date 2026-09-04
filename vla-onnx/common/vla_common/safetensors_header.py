"""Read a safetensors header without loading (or fully downloading) the weights.

safetensors stores its JSON index at the head of the file, so exact per-tensor shapes
and dtypes are readable from the first few KB. That is what makes it possible to size
each prospective TensorRT engine BEFORE building anything — on the 8 GB Orin the build
peak tracks the FP32 weight slice a single engine carries, so the split has to be
planned from the header, not discovered from a failed build.

How tensors are grouped into components is model-specific and stays in each playbook.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

# bytes per element, for the dtypes these checkpoints actually use
DTYPE_BYTES = {
    "F64": 8, "I64": 8,
    "F32": 4, "I32": 4,
    "F16": 2, "BF16": 2, "I16": 2,
    "I8": 1, "U8": 1, "BOOL": 1,
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    elements: int

    @property
    def bytes(self) -> int:
        return self.elements * DTYPE_BYTES.get(self.dtype, 4)


def read_header(path: str | Path) -> dict:
    """The raw JSON index. Accepts a full checkpoint or just its header range."""
    path = Path(path)
    with path.open("rb") as fh:
        raw_size = fh.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"{path} is too short to contain a safetensors header")
        (header_size,) = struct.unpack("<Q", raw_size)
        raw_header = fh.read(header_size)
    if len(raw_header) != header_size:
        raise ValueError(
            f"{path} contains {len(raw_header):,} of {header_size:,} required header bytes"
        )
    return json.loads(raw_header)


def read_tensors(path: str | Path) -> list[TensorInfo]:
    """Every tensor in the checkpoint, with `__metadata__` dropped."""
    out = []
    for name, meta in read_header(path).items():
        if name == "__metadata__":
            continue
        shape = tuple(meta["shape"])
        elements = 1
        for d in shape:
            elements *= d
        out.append(TensorInfo(name, shape, meta["dtype"], elements))
    return out


def total_params(tensors: list[TensorInfo]) -> int:
    return sum(t.elements for t in tensors)


def group_by(tensors: list[TensorInfo], classify) -> dict[str, int]:
    """Parameter count per group, where `classify(name) -> label` is model-specific."""
    out: dict[str, int] = {}
    for t in tensors:
        out[classify(t.name)] = out.get(classify(t.name), 0) + t.elements
    return out
