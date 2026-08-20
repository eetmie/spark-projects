# Episode-boundary pauses in masi_digging — trim proposal

**Status: BUILT 2026-08-20, not yet trained on.** `excavator/make_trim_variant.py` implements
this; `datasets/masi_digging_clean` (+ `_clean_ir`) are materialised. No training has used them
yet -- the running camera-count comparison (`outputs/digging189/{ir,both}`) stays on the
untrimmed data so it differs in exactly one variable. Train on the cleaned set as a separate
one-variable experiment against `outputs/digging189/ir`.

Two things were added beyond the original proposal: the cut points are derived per episode
from the signals rather than a fixed 30/15 window (see the script's docstring), and the 83-90
block was skipped outright as dirty -- those episodes idle more, barely use the boom lift
(18-44% active vs 50-65% either side), and in at least one case fail the task. Reviewed by
eye: 82 good / 83 fails, 90 weird / 91 good, matching the metric boundary exactly.

## The finding

Every episode begins with the operator stationary — recording starts, then movement starts —
and ends the same way. Measured on all 189 episodes, mean `|action|` averaged over the four
channels, against the middle-of-episode baseline:

| window | eps 0–81 (thirds) | eps 82–188 (full + corrections) |
|---|---|---|
| middle-of-episode baseline | 0.2499 | 0.2612 |
| first 0.5 s | **−89.1%** | **−89.2%** |
| first 1.0 s | −75.7% | −73.4% |
| first 1.5 s | −61.6% | −61.5% |
| last 0.5 s | −90.7% | −77.5% |
| last 1.0 s | −80.1% | −60.7% |

**Full-cycle recordings do not escape this.** The start-of-episode dead zone is identical in
both halves of the dataset (−89%). Recording full cycles instead of thirds reduces the
*density* of the artifact — one start-pause and one end-pause per cycle of task content
instead of three — but does not remove it. Only trimming removes it.

## Why it is worth fixing

**About 10% of training frames sit within 1.0 s of an episode boundary** (10 200 of 103 671
in the current 170-episode training split). For scale, the deliberately-recorded short
correction episodes are 1.36% of frames — the boundary artifact is seven times larger and
nobody chose it.

It is genuine label noise, not merely wasted data. At an episode start the camera sees an
ordinary mid-task scene — bucket somewhere, sand somewhere — paired with an action near
zero, while elsewhere in the dataset that same visual state is paired with real digging. The
model receives contradictory supervision on observations it cannot distinguish. The expected
symptom on the machine is hesitation or stalling mid-cycle. **If a deployed policy pauses
mid-task, suspect this first.**

Note that SmolVLA already masks *padded* actions out of the loss
(`modeling_smolvla.py:385`, `losses = losses * in_episode_bound`), so chunks running past the
end of an episode contribute no gradient. That protects the tail somewhat. It does **not**
protect the head: the first frames of an episode are fully supervised, real, and wrong.

## The proposal

A metadata-only trim, the same trick `excavator/make_camera_variant.py` already uses for
cameras — rewrite episode bounds, symlink `data/` and `videos/`, touch no video bytes.

| trim | frames removed | share of training set | episodes left under chunk 50 |
|---|---|---|---|
| **30 head / 15 tail (recommended)** | 7 650 | 7.4% | 0 |
| 45 head / 30 tail | 12 750 | 12.3% | **9** |

30/15 is the right tradeoff: it removes the worst of the dead zone (the first 1.0 s is −75%)
while leaving every episode above the 50-frame chunk length. The aggressive 45/30 variant
would push nine of the short correction episodes below one chunk, making them unusable.

Trim from the **head more than the tail** — the head is fully supervised, the tail is
partially protected by the pad mask, and the tail dip is already smaller in the newer
full-cycle episodes (−77.5% vs −90.7%).

## How to test it honestly

Train IR-only on the trimmed variant with tonight's exact recipe (50 000 steps, batch 32,
chunk 50, lr 1e-4, seed 1000) and score it against `outputs/digging189/ir` on the **same**
held-out episodes. One variable. If the stall behaviour is real, the trimmed model should
beat the baseline on displacement error, and should visibly hesitate less when driven.

Score the held-out episodes **untrimmed** — the trim is a training-data fix, and the model
still has to handle whatever the robot actually presents at runtime.

## For future data collection

Both of these, not either:

1. Record full cycles rather than thirds — a third of the boundary events per unit of task.
2. Start recording *before* you start moving and trim the first second in post. This is the
   bigger win and it costs nothing, because it also works retroactively on data you already
   have.

---

*Measured 2026-08-20 against `~/Desktop/masi_digging` at 189 episodes / 113 781 frames.
Related: `NOTES.md`, `STATUS.md`, `excavator/make_camera_variant.py`.*
