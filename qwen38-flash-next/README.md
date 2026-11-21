# Qwen3.8 Flash-Next → Anthropic / OpenAI API

Based on [Salient Data’s vLLM recipe](https://github.com/salient-data/qwen38-next-dgx-recipe) and [hasso5703/alzeo’s Claude proxy](https://github.com/hasso5703/alzeo).

On a DGX Spark with NVIDIA Docker support:

```bash
cd qwen38-flash-next
./qwen prepare
QWEN_HOST=<server-ip> ./qwen start
./qwen status
```

API: `http://<server-ip>:30001/v1/messages` (Anthropic) or `/v1/chat/completions` (OpenAI).
Model: `qwen3.8-flash-next`. Key: `.local/config/api-key`.
Stop: `./qwen stop`. Without `QWEN_HOST`, the API binds to `127.0.0.1`.
