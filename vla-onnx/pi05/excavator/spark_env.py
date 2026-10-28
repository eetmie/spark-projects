"""DGX Spark (GB10, sm_121, 121 GB unified memory) JAX settings. Import before jax; each is
overridable from the environment.

- XLA's Triton GEMM crashes LLVM on sm_121 with bf16 ("Unsupported conversion from bf16 to f16");
  cuBLAS handles the matmuls instead.
- GPU and host share memory, and a checkpoint save copies arrays to host while the device copies
  stay allocated. The default BFC allocator keeps every freed block (the loaded base weights, the
  step's ~22 GB scratch) and needs the scratch contiguous: with EMA it grew to its whole cap and
  left too little for a save. cuda_async (CUDA's stream-ordered pool) avoids both: pi0.5 + EMA at
  batch 32 holds 72 GB, ~40 GB stays free while training and >= 14 GB through a save.
- 72 % cap, no preallocation. With BFC, 65 % cannot fit EMA's step and 85 % starves the host.
"""

import os
import pathlib

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.72")
os.environ.setdefault("HF_LEROBOT_HOME", str(pathlib.Path(__file__).resolve().parent.parent / "datasets"))
