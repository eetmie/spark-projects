# Attribution

Proxy and download/preflight helpers: [hasso5703/alzeo](https://github.com/hasso5703/alzeo),
commit `5cd7c89` / v1.42.0, MIT; see LICENSE. The proxy accepts
`QWEN38_API_KEY_FILE` and calls `client_compat.py` to translate Claude effort
aliases for the NVIDIA template. Other vendored helpers are unchanged.

Pinned vLLM image, runtime arguments, token map, model hashes and signing key:
[salient-data/qwen38-next-dgx-recipe](https://github.com/salient-data/qwen38-next-dgx-recipe),
commit `7432036e7d549ed266d16832c43ea960f2cb7d55`, Apache-2.0;
see `LICENSE.salient` and `NOTICE.salient`. The local controller uses
BF16 KV cache, FP32 recurrent state, two request slots, xhigh thinking, and
disables additional dense FP8 conversion. Build scripts, patches, benchmarks,
and published results from the upstream repository are omitted here.

Model weights and container components retain their publishers' licenses.
