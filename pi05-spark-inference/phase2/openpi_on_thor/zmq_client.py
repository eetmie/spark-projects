#!/usr/bin/env python3
"""ZeroMQ REQ client for the π0.5 policy server (``zmq_serve.py``).

Use ``ZmqPolicyClient`` from your robot control loop:

    client = ZmqPolicyClient(host="192.168.1.50", port=5555)
    out = client.infer(obs)          # obs: openpi observation dict
    actions = out["actions"]         # shape (action_horizon, action_dim)

Run as a script for a local smoke test against a running server:

    python phase2/openpi_on_thor/zmq_client.py --host 127.0.0.1 --runs 20
"""
import argparse
import time

import numpy as np
import zmq
from openpi_client import msgpack_numpy


class ZmqPolicyClient:
    """Synchronous REQ client. One ``infer`` call == one round trip."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout_ms: int = 60000):
        self._ctx = zmq.Context.instance()
        self._addr = f"tcp://{host}:{port}"
        self._timeout_ms = timeout_ms
        self._packer = msgpack_numpy.Packer()
        self._connect()

    def _connect(self):
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(self._addr)

    def _roundtrip(self, payload):
        self._sock.send(self._packer.pack(payload))
        try:
            return msgpack_numpy.unpackb(self._sock.recv())
        except zmq.Again:
            # On timeout a REQ socket is stuck mid-state-machine; rebuild it so
            # the caller can retry instead of deadlocking.
            self._sock.close(0)
            self._connect()
            raise TimeoutError(f"no reply from {self._addr} within {self._timeout_ms} ms")

    def get_metadata(self) -> dict:
        return self._roundtrip({"__cmd__": "metadata"})

    def ping(self) -> dict:
        return self._roundtrip({"__cmd__": "ping"})

    def infer(self, obs: dict) -> dict:
        out = self._roundtrip(obs)
        if isinstance(out, dict) and "error" in out and "actions" not in out:
            raise RuntimeError(f"server error:\n{out['error']}")
        return out


def _synthetic_libero_obs(prompt: str) -> dict:
    """A random LIBERO-shaped observation, for smoke testing only."""
    return {
        "observation/state": np.random.rand(8).astype(np.float32),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": prompt,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--prompt", default="pick up the black bowl and place it on the plate")
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    client = ZmqPolicyClient(args.host, args.port)
    print("server metadata:", client.get_metadata())

    roundtrip, server_infer = [], []
    actions = None
    for i in range(args.runs):
        obs = _synthetic_libero_obs(args.prompt)
        t = time.monotonic()
        out = client.infer(obs)
        dt = (time.monotonic() - t) * 1000
        roundtrip.append(dt)
        st = out.get("server_timing", {})
        server_infer.append(st.get("infer_ms", float("nan")))
        actions = out["actions"]
        print(f"  run {i + 1:2d}: roundtrip {dt:6.1f} ms   server_infer {server_infer[-1]:6.1f} ms")

    rt = np.array(roundtrip)
    si = np.array(server_infer)
    print(f"\nactions shape: {np.asarray(actions).shape}")
    print(f"roundtrip   mean {rt.mean():6.1f} ± {rt.std():4.1f} ms  (min {rt.min():.1f}, p90 {np.percentile(rt, 90):.1f})")
    print(f"server infer mean {np.nanmean(si):6.1f} ms")
    print(f"transport+serialize overhead ≈ {rt.mean() - np.nanmean(si):.1f} ms")


if __name__ == "__main__":
    main()
