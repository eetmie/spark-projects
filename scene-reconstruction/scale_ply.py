#!/usr/bin/env python3
"""Metric-scale a trained 3DGS PLY from triangulated AprilTags.

usage: scale_ply.py <dataset> <in.ply> <out.ply> <tag_edge_m>
<tag_edge_m> = black-square edge (the edge AprilTag detects). Needs <dataset>/markers.json
from `spirula marker detect`. Positions *= s, log-scales += ln s.
"""
import json, sys
import numpy as np

data, src, dst, edge_m = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
edges = []
for t in json.load(open(f"{data}/markers.json"))["markers"]:
    c = np.array(t["corners"], float)
    edges += [np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]
edges = np.array(edges)
s = edge_m / edges.mean()
print(f"{len(edges)//4} tags, edge spread {100*edges.std()/edges.mean():.2f}%, scale {s:.5f} m/unit")
assert edges.std() / edges.mean() < 0.02, "tag edges disagree >2%: check detections"

raw = open(src, "rb").read()
end = raw.index(b"end_header\n") + len(b"end_header\n")
props = [l.split() for l in raw[:end].splitlines() if l.startswith(b"property")]
assert all(p[1] == b"float" for p in props), "expects an all-float PLY"
v = np.frombuffer(raw, np.dtype([(p[-1].decode(), "<f4") for p in props]), offset=end).copy()
for k in "xyz":
    v[k] *= s
for k in ("scale_0", "scale_1", "scale_2"):
    v[k] += np.float32(np.log(s))
open(dst, "wb").write(raw[:end] + v.tobytes())
print(f"wrote {dst} ({len(v)} splats)")
