"""Resolve the SmolVLA logical input roles to an engine's actual tensor names.

The ONNX exported on the Spark and the one from the older bench project disagree
on names (`image` vs `image0`, `image_mask` vs `img_mask0`, ...). Rather than
hard-code either, we match each logical role to a real tensor by name keywords
first, then fall back to a dtype+rank signature. Dims (image size, language
length, state/action) are then read straight off the resolved tensors, so the
rest of the runtime auto-configures to whatever was baked into the graph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

LOG = logging.getLogger("smolvla_runtime.io_spec")


@dataclass
class TensorSpec:
    name: str
    np_dtype: np.dtype
    shape: tuple[int, ...]   # static dims; -1 for dynamic


# role -> (name keywords, predicate(dtype, rank))
def _is_float(dt):   return np.issubdtype(dt, np.floating)
def _is_int(dt):     return np.issubdtype(dt, np.integer)
def _is_bool(dt):    return dt == np.bool_

_ROLE_RULES = {
    # order matters: match the more specific (masks) before the generic floats
    "img_mask":    (("img", "image"), ("mask",), lambda dt, r: _is_bool(dt) and r <= 1),
    "lang_masks":  (("lang", "attention"), ("mask",), lambda dt, r: _is_bool(dt) and r == 2),
    "lang_tokens": (("lang", "token", "input_id"), (), lambda dt, r: _is_int(dt) and r == 2),
    "image":       (("image", "img", "pixel"), (), lambda dt, r: _is_float(dt) and r == 4),
    "noise":       (("noise",), (), lambda dt, r: _is_float(dt) and r == 3),
    "state":       (("state",), (), lambda dt, r: _is_float(dt) and r == 2),
}


@dataclass
class ResolvedIO:
    role_to_name: dict[str, str]
    primary_output: str
    image_size: int
    lang_max_len: int
    state_dim: int
    chunk_size: int
    action_dim: int


def resolve_io(inputs: list[TensorSpec], outputs: list[TensorSpec]) -> ResolvedIO:
    by_name = {t.name: t for t in inputs}
    role_to_name: dict[str, str] = {}
    used: set[str] = set()

    def rank(t: TensorSpec) -> int:
        return len(t.shape)

    # Pass 1 — name keyword + predicate
    for role, (incl, excl, pred) in _ROLE_RULES.items():
        for t in inputs:
            if t.name in used:
                continue
            lname = t.name.lower()
            if any(k in lname for k in incl) and all(k not in lname for k in excl) \
                    and pred(t.np_dtype, rank(t)):
                role_to_name[role] = t.name
                used.add(t.name)
                break

    # Pass 2 — fill remaining roles by dtype+rank predicate alone (unique match)
    for role, (_incl, _excl, pred) in _ROLE_RULES.items():
        if role in role_to_name:
            continue
        cands = [t for t in inputs if t.name not in used and pred(t.np_dtype, rank(t))]
        if len(cands) == 1:
            role_to_name[role] = cands[0].name
            used.add(cands[0].name)

    missing = [r for r in _ROLE_RULES if r not in role_to_name]
    if missing:
        raise RuntimeError(
            f"Could not map SmolVLA input roles {missing} to engine tensors "
            f"{[ (t.name, str(t.np_dtype), t.shape) for t in inputs ]}. "
            "The export interface may have changed — inspect and extend io_spec."
        )

    # primary output = the float, rank-3 tensor (the action chunk)
    out_cands = [t for t in outputs if _is_float(t.np_dtype) and rank(t) == 3]
    primary = (out_cands[0] if out_cands else outputs[0]).name

    img = by_name[role_to_name["image"]].shape
    lang = by_name[role_to_name["lang_tokens"]].shape
    state = by_name[role_to_name["state"]].shape
    noise = by_name[role_to_name["noise"]].shape

    resolved = ResolvedIO(
        role_to_name=role_to_name,
        primary_output=primary,
        image_size=int(img[2]) if img[2] > 0 else 512,
        lang_max_len=int(lang[1]) if lang[1] > 0 else 48,
        state_dim=int(state[1]) if state[1] > 0 else 32,
        chunk_size=int(noise[1]) if noise[1] > 0 else 50,
        action_dim=int(noise[2]) if noise[2] > 0 else 32,
    )
    LOG.info("Resolved I/O: %s  -> output %s", resolved.role_to_name, resolved.primary_output)
    LOG.info("Dims: image=%d lang=%d state=%d chunk=%d action=%d",
             resolved.image_size, resolved.lang_max_len, resolved.state_dim,
             resolved.chunk_size, resolved.action_dim)
    return resolved
