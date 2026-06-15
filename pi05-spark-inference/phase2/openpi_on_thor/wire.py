#!/usr/bin/env python3
"""Wire helpers for the π0.5 ZMQ transport: JPEG-on-wire obs + bind safety.

Two gaps the bare msgpack-numpy path leaves open:

1. **Camera frames cross the LAN as raw numpy.** A 224×224×3 uint8 image is
   ~150 KB on the wire, ×2 cameras, every tick. JPEG-compressing *only* the
   camera frames cuts that ~10-20×. We compress a value only when its key is a
   known camera key (``JPEG_WHITELIST``) AND it looks like an image (ndim==3,
   uint8). JPEG is lossy, so the key check is what stops us from silently
   corrupting a state vector or a mask that merely shares that shape. Decode is
   tag-driven, so the server understands both compressed and raw clients —
   adding this is backward compatible.

2. **The REP socket binds to the LAN with no auth.** ``assert_bind_allowed``
   refuses a non-loopback bind unless the operator explicitly passes
   ``--allow-lan``, so an open policy server doesn't end up on the lab network
   by accident.

The JPEG-whitelist idea is adapted from FastCrest Tether's ZMQ serializer
(itself ported from FluxVLA, Apache-2.0). This is a clean reimplementation
sized to our openpi msgpack-numpy contract, not a copy.

Note on colour order: cv2.imencode/imdecode are symmetric, so an RGB array
round-trips to the same channel positions (OpenCV's nominal "BGR" labelling
cancels out) — no cvtColor needed as long as both ends use cv2.
"""
from __future__ import annotations

import ipaddress
import logging
from typing import Any

import numpy as np
from openpi_client import msgpack_numpy

logger = logging.getLogger("wire")

# Camera keys in our openpi observation dicts — only these get JPEG-compressed.
# Mirrors the keys the client/robot loop sends (see ZMQ.md).
JPEG_WHITELIST: frozenset[str] = frozenset({
    "observation/image",
    "observation/wrist_image",
})

_JPEG_TAG = "__jpeg__"
_warned_unwhitelisted: set[str] = set()
_warned_no_cv2 = False


def _cv2_or_none():
    """Import cv2 lazily; warn once and degrade to raw numpy if it's missing."""
    global _warned_no_cv2
    try:
        import cv2
        return cv2
    except ImportError:
        if not _warned_no_cv2:
            _warned_no_cv2 = True
            logger.warning("cv2 unavailable — images go over the wire uncompressed (raw numpy).")
        return None


def _should_jpeg(key: str, value: Any) -> bool:
    if not isinstance(value, np.ndarray) or value.ndim != 3 or value.dtype != np.uint8:
        return False
    if key in JPEG_WHITELIST:
        return True
    # Image-shaped but not whitelisted: send raw, warn once so it's noticed.
    if key not in _warned_unwhitelisted:
        _warned_unwhitelisted.add(key)
        logger.warning(
            "wire: key %r looks like an image (shape=%s, uint8) but isn't in "
            "JPEG_WHITELIST — sending raw (~%d KB). Add the key to compress it.",
            key, value.shape, value.nbytes // 1024,
        )
    return False


def encode_obs(
    obs: dict[str, Any],
    packer: msgpack_numpy.Packer,
    *,
    jpeg_quality: int | None = 85,
) -> bytes:
    """msgpack-pack an observation, JPEG-compressing whitelisted camera frames.

    Pass ``jpeg_quality=None`` to disable compression entirely (e.g. for a
    bit-exact parity run, or when cv2 isn't installed). Non-image fields are
    packed by msgpack-numpy exactly as before. Control messages (``__cmd__``)
    carry no image keys, so they pass straight through.
    """
    cv2 = _cv2_or_none() if jpeg_quality is not None else None
    if cv2 is None:
        return packer.pack(obs)

    out: dict[str, Any] = {}
    for key, value in obs.items():
        if _should_jpeg(key, value):
            ok, buf = cv2.imencode(".jpg", value, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
            if not ok:
                raise RuntimeError(f"JPEG encode failed for {key!r}")
            out[key] = {_JPEG_TAG: True, "data": buf.tobytes(), "shape": list(value.shape)}
        else:
            out[key] = value
    return packer.pack(out)


def decode_obs(raw: bytes) -> Any:
    """msgpack-unpack an observation, restoring any JPEG-compressed frames.

    Tag-driven, so a raw (uncompressed) payload decodes unchanged — backward
    compatible with clients that don't compress.
    """
    msg = msgpack_numpy.unpackb(raw)
    if not isinstance(msg, dict):
        return msg
    cv2 = None
    for key, value in msg.items():
        if isinstance(value, dict) and value.get(_JPEG_TAG):
            cv2 = cv2 or _cv2_or_none()
            if cv2 is None:
                raise RuntimeError(
                    f"received JPEG-compressed {key!r} but cv2 is unavailable to decode it"
                )
            arr = np.frombuffer(value["data"], dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"JPEG decode failed for {key!r}")
            msg[key] = img
    return msg


# --- bind safety -------------------------------------------------------------
def is_loopback_bind(host: str) -> bool:
    """True when a ZMQ bind host is local-only (not reachable from the LAN)."""
    h = host.strip().strip("[]")
    if h == "localhost":
        return True
    if h in {"", "*", "0.0.0.0", "::"}:
        return False
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def assert_bind_allowed(host: str, *, allow_lan: bool) -> None:
    """Refuse a LAN-reachable bind unless the operator explicitly allows it.

    The server has no authentication, so a non-loopback bind exposes an open
    policy endpoint — anyone on the network could drive the robot. Require an
    explicit ``--allow-lan`` acknowledgment (trusted/isolated lab network only),
    or bind to 127.0.0.1 for local use.
    """
    if is_loopback_bind(host):
        return
    if allow_lan:
        logger.warning(
            "ZMQ server bound to %s with NO authentication — anyone on the LAN "
            "can drive the robot. Use only on a trusted/isolated network; for an "
            "untrusted link use a CurveZMQ keypair or an SSH tunnel.",
            host,
        )
        return
    raise SystemExit(
        f"Refusing to bind the ZMQ policy server to {host!r}: it has no "
        "authentication and would be reachable from the LAN. Re-run with "
        "--allow-lan to accept this (trusted lab network only), or use "
        "--host 127.0.0.1 for local-only serving."
    )


__all__ = [
    "JPEG_WHITELIST",
    "encode_obs",
    "decode_obs",
    "is_loopback_bind",
    "assert_bind_allowed",
]
