"""Run an OpenPI script with the excavator configs registered and DGX Spark settings applied.

    python excavator/run.py train pi05_excavator --exp-name smoke [tyro overrides...]
    python excavator/run.py norm-stats --config-name pi05_excavator

JAX settings are in spark_env.py. Checkpoint saving on the Spark's 121 GB unified memory:
- Orbax otherwise writes the whole checkpoint as one in-memory OCDBT transaction, a second full
  copy until commit; a byte limit streams it (OPENPI_SAVE_CONCURRENT_GB, default 5, >= the largest
  array, 4.8 GB fp32).
- A full save's host copy adds ~40 GB (~53 GB with EMA), so checkpoints keep params (the EMA copy
  when EMA is on) + norm stats only unless OPENPI_SAVE_OPT_STATE=1.
"""

import functools
import os
import pathlib
import runpy
import sys

HERE = pathlib.Path(__file__).resolve().parent
OPENPI = HERE.parent / "third_party" / "openpi"
SCRIPTS = {"train": "train.py", "norm-stats": "compute_norm_stats.py"}

sys.path.insert(0, str(HERE))
import spark_env  # noqa: E402, F401
import orbax.checkpoint as ocp  # noqa: E402

ocp.PyTreeCheckpointHandler = functools.partial(
    ocp.PyTreeCheckpointHandler, save_concurrent_gb=int(os.environ.get("OPENPI_SAVE_CONCURRENT_GB", "5"))
)

from openpi.training import checkpoints as _checkpoints  # noqa: E402

if os.environ.get("OPENPI_SAVE_OPT_STATE", "0") != "1":
    # Params + norm stats only (~13 GB host copy instead of ~40 GB with AdamW state). Enough for
    # export and inference; --resume then has no optimizer state, so it is not supported.
    def _save_params_only(checkpoint_manager, state, data_loader, step):
        def save_assets(directory):
            data_config = data_loader.data_config()
            if data_config.norm_stats is not None and data_config.asset_id is not None:
                _checkpoints._normalize.save(directory / data_config.asset_id, data_config.norm_stats)

        with _checkpoints.at.disable_typechecking():
            _, params = _checkpoints._split_params(state)
        checkpoint_manager.save(step, {"assets": save_assets, "params": {"params": params}})

    _checkpoints.save_state = _save_params_only

import config  # noqa: E402

config.register()

if len(sys.argv) < 2 or sys.argv[1] not in SCRIPTS:
    sys.exit(f"usage: run.py {{{','.join(SCRIPTS)}}} [args...]")
script = OPENPI / "scripts" / SCRIPTS[sys.argv[1]]
sys.argv = [str(script), *sys.argv[2:]]
runpy.run_path(str(script), run_name="__main__")
