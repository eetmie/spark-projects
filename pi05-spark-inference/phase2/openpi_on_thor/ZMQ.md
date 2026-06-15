# ZeroMQ policy serving (π0.5 over LAN)

A minimal REQ/REP transport so a robot can drive the GB10-hosted π0.5 policy over
the network. Wraps `policy.infer()` — same contract as openpi's websocket server
(msgpack-numpy payloads, `server_timing` on replies) but synchronous REQ/REP,
which is the natural fit for **chunked** inference: one observation in, one
action chunk (`action_horizon × action_dim`) out.

- `zmq_serve.py` — REP server (PyTorch BF16 or TensorRT engine).
- `zmq_client.py` — `ZmqPolicyClient` + a local smoke test.

## Why REQ/REP (and not a high-rate video stream)

With action chunking the model only needs **one observation per inference**
(~10 Hz), not a continuous video feed. The robot replays the 10-action chunk
locally at its control rate, so inference latency hides between chunks. One
LIBERO obs = 2×(224×224×3) images + an 8-vector ≈ 300 KB raw → ~24 Mbit/s at
10 Hz, trivial on gigabit LAN (~2–3 ms transfer, hidden inside the ~95 ms infer).

**JPEG-on-wire (default on).** The client JPEG-compresses the two camera frames
before sending (`wire.encode_obs`, quality 85), cutting that ~300 KB to ~20–40 KB
— a non-issue on gigabit, but it matters on Wi-Fi / congested links and leaves
headroom if you raise camera resolution. Compression is keyed to a whitelist
(`observation/image`, `observation/wrist_image`) so only real frames are touched
— JPEG is lossy, so a state vector that happened to be uint8 HxWx3 is never
compressed. Decode is tag-driven, so the server still accepts raw clients. For a
**lossless** payload (e.g. a reference comparison) construct the client with
`ZmqPolicyClient(..., jpeg_quality=None)`.

## Run it

Server (on the GB10 box). TensorRT backend (~95 ms); drop `--engine-path` for
PyTorch BF16 (~200 ms):

```bash
DR='docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v '"$PWD"':/workspace -v '"$PWD"'/.cache:/cache -w /workspace \
    -e PYTHONPATH=/workspace/phase2 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -p 5555:5555 pi05-spark-trt:latest'

$DR python phase2/openpi_on_thor/zmq_serve.py \
    --config-name pi05_libero \
    --checkpoint-dir /workspace/checkpoints/pi05_libero_pytorch \
    --engine-path /workspace/checkpoints/pi05_libero_pytorch/onnx/model_fp8_nvfp4.engine \
    --host 0.0.0.0 --allow-lan \
    --port 5555
```

> Note the added `-p 5555:5555` to publish the port out of the container.
>
> The server now **binds to `127.0.0.1` by default** (local-only). To serve a
> robot over the LAN you must pass `--host 0.0.0.0 --allow-lan` — the explicit
> `--allow-lan` is a deliberate acknowledgment that the endpoint is
> unauthenticated (see *Scope / limits*). Without it, a non-loopback bind is
> refused.

Client (robot box, or same box for a smoke test):

```bash
python phase2/openpi_on_thor/zmq_client.py --host <server-ip> --port 5555 --runs 20
```

## Use from your robot loop

```python
from openpi_on_thor.zmq_client import ZmqPolicyClient

client = ZmqPolicyClient(host="192.168.1.50", port=5555)
meta = client.get_metadata()          # {'action_horizon': 10, 'action_dim': 32, ...}

while running:
    obs = {
        "observation/image":       base_cam_uint8_hwc,    # (224,224,3) uint8
        "observation/wrist_image": wrist_cam_uint8_hwc,   # (224,224,3) uint8
        "observation/state":       robot_state,           # (8,) float
        "prompt":                  "pick up the black bowl",
    }
    chunk = client.infer(obs)["actions"]   # (action_horizon, action_dim)
    for a in chunk:
        robot.apply(a)                     # execute open-loop within the chunk
```

## Protocol

- Observation request → reply `{"actions": ndarray, "policy_timing": {...}, "server_timing": {"infer_ms": ...}}`.
- Control: send `{"__cmd__": "metadata"}` or `{"__cmd__": "ping"}`.
- Errors come back as `{"error": "<traceback>"}` (REP always replies, so the
  socket never deadlocks); the client raises `RuntimeError`.

## Scope / limits

- **One client, strict lockstep** (REQ/REP alternates send→recv). Perfect for a
  single robot. For multiple robots or async pipelining, switch the server to
  `zmq.ROUTER` and the client to `zmq.DEALER` (envelope frames, no lockstep).
- No auth/encryption — intended for a trusted LAN. A non-loopback bind is
  refused unless you pass `--allow-lan` (which logs a loud warning), so you can't
  expose it by accident. Use a CurveZMQ keypair or an SSH tunnel if it must cross
  an untrusted network.
- Client `infer()` has a 60 s receive timeout; on timeout it rebuilds the socket
  and raises `TimeoutError` so your loop can retry.
