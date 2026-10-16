"""DGX Spark (GB10, sm_121, 121 GB unified memory) JAX settings. Import before jax; each is
overridable from the environment.

- XLA's Triton GEMM crashes LLVM on sm_121 with bf16 ("Unsupported conversion from bf16 to f16");
  cuBLAS handles the matmuls instead.
- No preallocation and a 65 % cap: GPU and host share memory, and a checkpoint save copies arrays
  to host while the device copies stay allocated.
"""

import os
import pathlib

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.65")
os.environ.setdefault("HF_LEROBOT_HOME", str(pathlib.Path(__file__).resolve().parent.parent / "datasets"))
