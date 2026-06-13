#!/usr/bin/env bash
# Put the Orin Nano Super into its fastest, most deterministic state:
#   - nvpmodel mode 2 = MAXN_SUPER  (no power cap)
#   - jetson_clocks               = pin every clock to max (no DVFS ramp)
#
# This is the one-shot, run-it-now version. For persistence across reboots,
# install system/jetson-perf.service instead (see README).
#
# JetPack 7.2 / L4T R39.2 nvpmodel IDs on this board:
#   0 = 15W   1 = 25W   2 = MAXN_SUPER     (verify with: sudo nvpmodel -p --verbose)
set -euo pipefail

MODE="${1:-2}"   # default MAXN_SUPER

echo ">> Setting nvpmodel -m ${MODE} (MAXN_SUPER) ..."
sudo nvpmodel -m "${MODE}"

echo ">> Pinning clocks (jetson_clocks) ..."
sudo jetson_clocks

echo
echo ">> Current power mode:"
sudo nvpmodel -q
echo
echo ">> Pinned clocks:"
sudo jetson_clocks --show | head -n 20
