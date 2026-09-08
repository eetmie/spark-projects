# Retired eval presets (2026-09-08)

`eval_compare.py` used to carry a `PRESETS` table: per experiment, the source recording,
a pinned instruction, and a hand-written list of held-out episodes. It was deleted when
the eval moved to deriving all of that from the checkpoint's own `train_config.json`.

This file exists because for three of those recordings the preset was **the last surviving
record of what a sweep held out** — `smolvla/outputs/` and `xvla/outputs/` have since been
cleared, so the checkpoints those splits belong to are gone and the lists cannot be
re-derived from anything. Kept as provenance for numbers quoted in `STATUS.md`,
`notes/`, and the article draft. Not runnable, not maintained.

| preset | source | held-out episodes | instruction |
|---|---|---|---|
| `kaivuri` | `Desktop/vanhat/masi_kaivuri_juusto` (31 eps) | `3 11 19 27` | `scoop blocks and dump it to the left` |
| `digging` | `Desktop/masi_digging` @ 82 eps | `5 15 25 35 45 55 65 75` | `move the sand to the container` |
| `digging189` | `Desktop/masi_digging` @ 189 eps | `range(5, 189, 10)` (19) | `move the sand to the container` |
| `digging_clean` | `masi_digging_clean` (181 eps) | `5 15 25 35 45 55 65 75 87 97 107 117 127 137 147 157 167 177` | pinned `move the sand to the container` |
| `digging_dry` | `Desktop/masi_digging_dry` (62 eps) | `range(5, 62, 10)` (6) | `move sand to container` |
| `digging_dry2` | `Desktop/masi_digging_dry_2` @ 78 eps | `range(5, 78, 10)` (8) | per-episode |
| `digging_dry2_sand` | same | `5 15 25 35 45 55` | per-episode |
| `digging_dry2_rock` | same | `65 75` | per-episode |

## Why they were retired rather than repaired

Every field of the table drifted from the data it described, silently, because nothing
tied the two together:

- **`digging_dry2` under-scored by 3x.** Written when the recording had 78 episodes; it
  grew to 242. The trainer's own rule holds out 24, the preset scored 8 of them, and the
  result was reported as the model's error. Every published dry2 number was computed on a
  third of the held-out set. A new table on the same checkpoint will differ — that is the
  fix working, not a regression.
- **`digging_dry2`'s block comment went stale.** It claimed "eps 0-62 sand, 63-77 rock".
  The recording is now eight interleaved blocks, 154 sand / 88 rock. The rock slice's
  `65 75` still landed on rock purely by coincidence, and covered 2 of the 10 rock
  episodes actually held out.
- **`digging_clean` was live train/eval leakage.** Ten of its eighteen "held-out" ids
  (`87 97 … 177`) are episodes the trainer's `range(5, 181, 10)` rule puts in the
  TRAINING set. It was defused only because `queue_digging_clean.sh` happened to export a
  matching `VAL_EPISODES`; invoking the trainer directly leaked the eval set with no error
  and a plausible loss curve.
- **`digging_clean` also pinned the wrong instruction.** It said
  `move the sand to the container`; the dataset says `move sand to container` — the trim
  builder normalized the string and the preset was never updated. The policy conditions on
  the language embedding, so it was scored on an instruction it never trained on. Nothing
  errors; the prefix is well-formed and the numbers come out plausible.
- **`digging` and `digging189`** pointed at the identical 189-episode directory and
  differed only in their held-out list; the 82-episode recording no longer exists.

## What replaced it

`eval_compare.py --ckpt <checkpoint>` reads `train_config.json`: `dataset.episodes` is
what the run trained on, so the held-out set is its complement — derived, never typed.
`dataset.root` resolved through its `data/` symlink gives the source recording, and the
instructions come from that recording's `meta/tasks.parquet`, per episode. The sand/rock
decomposition is `--only-task "move rock to container"`, which finds all ten.
