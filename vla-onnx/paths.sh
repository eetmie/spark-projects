#!/usr/bin/env bash
# Shared path resolution for the vla-onnx pipeline. Source it, never run it:
#
#     HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
#     source "$HERE/../../paths.sh"
#
# Everything resolves from this file's own location, so a clone works anywhere and
# a moved tree does not need a sed pass. Override any of them from the environment.
#
# Why this exists: the tree has already been renamed twice (~/GitHub -> ~, then the
# 2026-09 restructure), and each time a crop of absolute paths went stale silently —
# a script would run and write to a directory nobody was reading.

VLA_ONNX=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SPARK_PROJECTS=$(cd "$VLA_ONNX/.." && pwd)

# The two pipeline environments. lerobot 0.5.1 is the comparison-pinned venv that
# smolvla and xvla fine-tune in; 0.6.1 is what evo1 needs and what the X-VLA export
# runs in. They are deliberately NOT interchangeable — see vla-onnx/README.md.
VENV_LEROBOT051=${VENV_LEROBOT051:-$VLA_ONNX/.venv-lerobot051}
VENV_LEROBOT061=${VENV_LEROBOT061:-$VLA_ONNX/.venv-lerobot061}

# Recorded excavator datasets. Not in git and not derived — the only copy of the
# source recordings. Point VLA_DATASETS elsewhere to run from another disk.
VLA_DATASETS=${VLA_DATASETS:-$HOME/Desktop}

# Where finished bundles are collected before shipping to the robot.
VLA_BUNDLES=${VLA_BUNDLES:-$HOME/bundles}

export VLA_ONNX SPARK_PROJECTS VENV_LEROBOT051 VENV_LEROBOT061 VLA_DATASETS VLA_BUNDLES
