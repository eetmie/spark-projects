#!/usr/bin/env python3
"""ZeroMQ REQ/REP policy server for π0.5 (openpi) on DGX Spark / GB10.

Wraps ``policy.infer()`` behind a ZMQ REP socket so a remote robot client can
request action chunks over LAN. Mirrors the openpi ``WebsocketPolicyServer``
contract (msgpack-numpy payloads, ``server_timing`` on the reply) but uses a
synchronous REQ/REP pattern that fits chunked inference: one observation in,
one action chunk out.

Backends:
  * PyTorch BF16  -- omit ``--engine-path``
  * TensorRT FP8+NVFP4 -- pass ``--engine-path .../model_fp8_nvfp4.engine``

Client: see ``zmq_client.py``.
"""
import argparse
import logging
import time
import traceback

import zmq
from openpi_client import msgpack_numpy
from openpi.policies import policy_config
from openpi.training import config as _config

# ---------------------------------------------------------------------------
# load_pytorch patch (dtype / tied-weight tolerant) -- identical to the one in
# pi05_inference.py, replicated here so this server does not import lerobot.
# ---------------------------------------------------------------------------
import safetensors.torch as _st
from openpi.models_pytorch import pi0_pytorch as _pi0pt
import openpi.models.model as _model_mod


def _load_pytorch_patched(self, train_config, weight_path: str):
    model = _pi0pt.PI0Pytorch(config=train_config.model)
    model.load_state_dict(_st.load_file(weight_path), strict=False)
    return model


for _cls in vars(_model_mod).values():
    if isinstance(_cls, type) and hasattr(_cls, "load_pytorch"):
        _cls.load_pytorch = _load_pytorch_patched
# ---------------------------------------------------------------------------

logger = logging.getLogger("zmq_serve")


def build_policy(config_name: str, checkpoint_dir: str, engine_path: str | None):
    config = _config.get_config(config_name)
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    if engine_path:
        from openpi_on_thor.trt_model_forward import setup_pi0_tensorrt_engine

        policy = setup_pi0_tensorrt_engine(policy, engine_path)
        logger.info("TensorRT engine attached: %s", engine_path)
    return policy, config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-name", default="pi05_libero")
    ap.add_argument("--checkpoint-dir", default="/workspace/checkpoints/pi05_libero_pytorch")
    ap.add_argument("--engine-path", default=None,
                    help="TensorRT engine path; omit to serve PyTorch BF16")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--warmup", type=int, default=3,
                    help="synthetic warmup infers before accepting requests")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    backend = "tensorrt" if args.engine_path else "pytorch_bf16"
    logger.info("Loading policy (%s)...", backend)
    policy, config = build_policy(args.config_name, args.checkpoint_dir, args.engine_path)

    metadata = {
        "config": args.config_name,
        "backend": backend,
        "action_horizon": int(config.model.action_horizon),
        "action_dim": int(config.model.action_dim),
    }

    # Warm up with a synthetic example so the first real request isn't slow
    # (compile / cudnn autotune / TRT first-launch all happen here).
    if args.warmup > 0:
        from openpi.policies.libero_policy import make_libero_example

        ex = make_libero_example()
        logger.info("Warming up (%d runs)...", args.warmup)
        for i in range(args.warmup):
            t = time.monotonic()
            policy.infer(ex)
            logger.info("  warmup %d/%d  %.1f ms", i + 1, args.warmup, (time.monotonic() - t) * 1000)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{args.host}:{args.port}")
    logger.info("ZMQ REP server ready on tcp://%s:%d  (backend=%s)", args.host, args.port, backend)

    packer = msgpack_numpy.Packer()
    prev_total = None
    n = 0
    while True:
        raw = sock.recv()  # REP must reply exactly once per recv()
        start = time.monotonic()
        try:
            msg = msgpack_numpy.unpackb(raw)

            # Control messages: {"__cmd__": "metadata" | "ping"}
            if isinstance(msg, dict) and "__cmd__" in msg:
                cmd = msg["__cmd__"]
                if cmd == "metadata":
                    sock.send(packer.pack(metadata))
                elif cmd == "ping":
                    sock.send(packer.pack({"pong": True}))
                else:
                    sock.send(packer.pack({"error": f"unknown cmd {cmd!r}"}))
                continue

            infer_t = time.monotonic()
            action = policy.infer(msg)
            infer_t = time.monotonic() - infer_t

            action["server_timing"] = {"infer_ms": infer_t * 1000}
            if prev_total is not None:
                action["server_timing"]["prev_total_ms"] = prev_total * 1000
            sock.send(packer.pack(action))

            n += 1
            prev_total = time.monotonic() - start
            logger.info("req %d: infer %.1f ms (total %.1f ms)", n, infer_t * 1000, prev_total * 1000)

        except Exception:
            tb = traceback.format_exc()
            logger.error("infer failed:\n%s", tb)
            # A REP socket must always send a reply, otherwise its state machine
            # locks up and the next recv() will assert. Reply with the error.
            sock.send(packer.pack({"error": tb}))


if __name__ == "__main__":
    main()
