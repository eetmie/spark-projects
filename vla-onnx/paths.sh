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

# The X-VLA environment (lerobot 0.6.1): fine-tune, eval and export.
VENV_LEROBOT061=${VENV_LEROBOT061:-$VLA_ONNX/.venv-lerobot061}

# Recorded excavator datasets. Not in git and not derived — the only copy of the
# source recordings. Point VLA_DATASETS elsewhere to run from another disk.
VLA_DATASETS=${VLA_DATASETS:-$HOME/Desktop}

# Where finished bundles are collected before shipping to the robot.
VLA_BUNDLES=${VLA_BUNDLES:-$HOME/bundles}

# jetson-orin-nano-vla checkout: its export.sh turns a checkpoint into the Orin bundle.
JETSON_VLA=${JETSON_VLA:-$HOME/jetson-orin-nano-vla}

export VLA_ONNX SPARK_PROJECTS VENV_LEROBOT061 VLA_DATASETS VLA_BUNDLES JETSON_VLA
