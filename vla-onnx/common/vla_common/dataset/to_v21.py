#!/usr/bin/env python
"""Write a LeRobot v2.1 copy of a LeRobot v3.0 dataset, losslessly, and prove it.

WHY v2.1. The recorders write v3.0 (lerobot 0.5.1 and 0.6.1 both do), and the LeRobot
policies train on that directly. openpi does not: its `pyproject.toml` pins lerobot at
0cf8648 (2025-05-28), whose `CODEBASE_VERSION` is "v2.1", and it cannot read v3 at all
(openpi #818 asks for an upgrade; PR #972 only makes the failure readable). GR00T
N1.6/N1.7 also read v2 (Isaac-GR00T getting_started/data_preparation.md). lerobot ships
only the forward converter, `convert_dataset_v21_to_v30.py`, so this is the other half:
keep recording v3, derive a v2.1 copy for the trainers that need it.

WHAT CHANGES. Layout only; every value is the source's.

    v3.0                                         v2.1
    data/chunk-C/file-F.parquet (many eps)   ->  data/chunk-C/episode_E.parquet (one ep)
    videos/K/chunk-C/file-F.mp4 (many eps)   ->  videos/chunk-C/K/episode_E.mp4 (one ep)
    meta/episodes/chunk-C/file-F.parquet     ->  meta/episodes.jsonl + episodes_stats.jsonl
    meta/tasks.parquet                       ->  meta/tasks.jsonl

Data parquet is filtered with pyarrow, so column types survive byte-for-byte. Global
`index`, per-episode `frame_index` and `timestamp` are left alone: v3 already writes
them the way v2.1 reads them. One thing in the file does have to change: the
`huggingface` schema entry. lerobot 0.6.1 writes through datasets 4, which spells a
fixed-length vector `List`; openpi's lock resolves datasets 3.6.0, which only knows
`Sequence` and refuses the whole file. Copying the parquet as-is (as NVIDIA's
convert_v3_to_v2.py does) therefore produces a v2.1 dataset openpi cannot open. See
`legacy_schema`.

VIDEO IS COPIED, NOT RE-ENCODED. v3 encodes every episode on its own and joins them with
a stream copy (`concatenate_video_files`), so each episode starts on a keyframe and its
packets are contiguous. Cutting those packets back out and rebasing their timestamps to
zero is exact; re-encoding would add a second generation of compression loss to every
frame. Episodes are located by frame count, never by the float `from_timestamp`: packets
are ranked by pts and assigned to episodes in `from_timestamp` order using `length`.
NVIDIA's convert_v3_to_v2.py instead runs `ffmpeg -ss <from> -i ... -c copy`, which with
input seeking and `-avoid_negative_ts` can start a frame off or shift the timestamps --
the two cameras and the parquet rows must stay aligned to the frame. An episode that does
not start on a keyframe (a v3 written by some other tool) is re-encoded with the source's
codec and pixel format instead, and the report says so.

STATS. Per-episode stats come from the `stats/*` columns of meta/episodes. The pinned
lerobot aggregates them at load and asserts image stats are (3,1,1) and `count` is (1,)
(compute_stats._assert_type_and_shape), so image stats are reshaped to that. Quantiles
ride along; v2.1 ignores them, and openpi computes its own norm stats anyway
(`scripts/compute_norm_stats.py`, on this v2.1 copy).

VERIFIED BY DEFAULT. After writing, every episode's parquet rows are compared with the
source rows, and every frame of every output video is decoded and compared with the same
frame decoded from the source file (hash of the pixels), with its timestamp checked
against frame_index / fps. `--no-verify` skips the decode pass.

Everything else rides along: extra files under meta/ (a recorder's contract, say) and
top-level directories such as the native recorder's `native/`.

Usage:
    python -m vla_common.dataset.to_v21 --src ~/datasets/<rec> --dst ~/datasets/<rec>_v21
    python -m vla_common.dataset.to_v21 --src <rec> --dst <dst> --no-verify

Needs pyarrow, numpy and PyAV (`av`); a lerobot 0.6.1 environment has all three. It does
not import lerobot, so it runs in either pipeline venv.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import shutil
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

V21 = "v2.1"
V21_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V21_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
V3_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
V3_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
# v3-only sizing hints; v2.1 has one file per episode.
V3_ONLY_INFO = ("data_files_size_in_mb", "video_files_size_in_mb")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="LeRobot v3.0 dataset root")
    p.add_argument("--dst", type=Path, required=True, help="v2.1 root to create")
    p.add_argument("--no-verify", action="store_true",
                   help="skip decoding every output frame against the source")
    p.add_argument("--force", action="store_true", help="replace an existing --dst")
    return p.parse_args()


# --------------------------------------------------------------------------- metadata
def load_episodes(src: Path) -> list[dict]:
    """meta/episodes/*.parquet rows, sorted by episode_index, as plain Python values."""
    rows = []
    for path in sorted((src / "meta" / "episodes").rglob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    rows.sort(key=lambda r: r["episode_index"])
    if [r["episode_index"] for r in rows] != list(range(len(rows))):
        raise SystemExit("meta/episodes is not a contiguous 0..N-1 episode range")
    return rows


def load_tasks(src: Path) -> list[tuple[int, str]]:
    table = pq.read_table(src / "meta" / "tasks.parquet")
    df = table.to_pandas()
    # v3 writes the task string as the index; tolerate it arriving as a column.
    if "task" not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or "index": "task"})
    return sorted((int(i), str(t)) for t, i in zip(df["task"], df["task_index"]))


def episode_stats(record: dict, features: dict) -> dict:
    """The `stats/<feature>/<stat>` columns of one episode row, nested and shaped for v2.1."""
    stats: dict[str, dict] = defaultdict(dict)
    for key, value in record.items():
        if not key.startswith("stats/") or value is None:
            continue
        _, feature, stat = key.split("/", 2)
        arr = np.asarray(value, dtype=np.float64 if stat != "count" else np.int64)
        if stat == "count":
            arr = arr.reshape(1)
        elif features.get(feature, {}).get("dtype") in ("video", "image"):
            arr = arr.reshape(3, 1, 1)
        stats[feature][stat] = arr.tolist()
    return dict(stats)


def convert_info(info: dict, n_episodes: int, video_keys: list[str]) -> dict:
    out = {k: v for k, v in info.items() if k not in V3_ONLY_INFO}
    out["codebase_version"] = V21
    out["data_path"] = V21_DATA_PATH
    out["video_path"] = V21_VIDEO_PATH if video_keys else None
    chunks = info.get("chunks_size", 1000)
    out["chunks_size"] = chunks
    out["total_chunks"] = math.ceil(n_episodes / chunks) if n_episodes else 0
    out["total_videos"] = n_episodes * len(video_keys)
    out["splits"] = info.get("splits") or {"train": f"0:{n_episodes}"}
    return out


# --------------------------------------------------------------------------- data
def _legacy_hf_features(node):
    """datasets 4 `List` -> datasets 3 `Sequence`, the same shape spelled the old way."""
    if isinstance(node, list):
        return [_legacy_hf_features(n) for n in node]
    if not isinstance(node, dict):
        return node
    out = {k: _legacy_hf_features(v) for k, v in node.items()}
    if out.get("_type") == "List":
        inner = out.get("feature")
        if isinstance(inner, dict) and "_type" not in inner:
            # datasets 3 reads Sequence(dict) as a dict of lists: not the same thing.
            raise SystemExit("a list-of-dicts feature has no datasets 3 equivalent")
        out["_type"] = "Sequence"
        out.setdefault("length", -1)
    return out


def legacy_schema(table: pa.Table) -> pa.Table:
    """Rewrite the parquet's `huggingface` schema entry so datasets 3.x can read it.

    lerobot 0.6.1 writes through datasets 4, which records fixed-length vectors as
    `{"_type": "List", "length": n}`. openpi's lock resolves datasets 3.6.0, which only
    knows `Sequence` and refuses the file outright ("Feature type 'List' not found").
    The arrow column is already a fixed_size_list -- what `Sequence(length=n)` reads --
    so only the metadata changes, never a value.
    """
    meta = dict(table.schema.metadata or {})
    if b"huggingface" not in meta:
        return table
    hf = json.loads(meta[b"huggingface"])
    if "info" in hf and "features" in hf["info"]:
        hf["info"]["features"] = _legacy_hf_features(hf["info"]["features"])
    meta[b"huggingface"] = json.dumps(hf).encode()
    return table.replace_schema_metadata(meta)


def convert_data(src: Path, dst: Path, episodes: list[dict], chunks: int) -> int:
    by_file = defaultdict(list)
    for ep in episodes:
        by_file[(ep["data/chunk_index"], ep["data/file_index"])].append(ep)
    frames = 0
    for (chunk, file), eps in sorted(by_file.items()):
        table = pq.read_table(src / V3_DATA_PATH.format(chunk_index=chunk, file_index=file))
        for ep in eps:
            part = table.filter(pc.equal(table["episode_index"], ep["episode_index"]))
            if part.num_rows != ep["length"]:
                raise SystemExit(f"episode {ep['episode_index']}: {part.num_rows} data rows, "
                                 f"meta says {ep['length']}")
            out = dst / V21_DATA_PATH.format(episode_chunk=ep["episode_index"] // chunks,
                                             episode_index=ep["episode_index"])
            out.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(legacy_schema(part), out)
            frames += part.num_rows
    return frames


# --------------------------------------------------------------------------- video
def _segments(eps: list[dict], key: str) -> list[dict]:
    """The episodes stored in one v3 video file, in their order inside it."""
    return sorted(eps, key=lambda e: float(e[f"videos/{key}/from_timestamp"]))


def _video_pts(path: Path) -> list[int]:
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return sorted(p.pts for p in container.demux(stream) if p.dts is not None)


def _reencode(src_file: Path, out: Path, first: int, count: int, fps: int) -> None:
    """Fallback for an episode that does not start on a keyframe: decode its frames and
    encode them again with the source codec and pixel format."""
    import av
    with av.open(str(src_file)) as inp:
        stream = inp.streams.video[0]
        codec, pix_fmt = stream.codec_context.name, stream.codec_context.pix_fmt
        with av.open(str(out), "w") as container:
            ost = container.add_stream(codec, rate=fps)
            ost.width, ost.height, ost.pix_fmt = stream.width, stream.height, pix_fmt
            ost.time_base = Fraction(1, fps)
            for i, frame in enumerate(inp.decode(stream)):
                if i < first:
                    continue
                if i >= first + count:
                    break
                frame = frame.reformat(format=pix_fmt)
                frame.pts, frame.time_base = i - first, Fraction(1, fps)
                for packet in ost.encode(frame):
                    container.mux(packet)
            for packet in ost.encode():
                container.mux(packet)


def convert_videos(src: Path, dst: Path, episodes: list[dict], video_keys: list[str],
                   chunks: int, fps: int) -> list[str]:
    import av
    report = []
    for key in video_keys:
        by_file = defaultdict(list)
        for ep in episodes:
            by_file[(ep[f"videos/{key}/chunk_index"], ep[f"videos/{key}/file_index"])].append(ep)
        for (chunk, file), eps in sorted(by_file.items()):
            src_file = src / V3_VIDEO_PATH.format(video_key=key, chunk_index=chunk, file_index=file)
            segments = _segments(eps, key)
            pts = _video_pts(src_file)
            if len(pts) != sum(e["length"] for e in segments):
                raise SystemExit(f"{src_file}: {len(pts)} frames, episodes say "
                                 f"{sum(e['length'] for e in segments)}")
            # Frame ranges per episode, by count: [start, end) in pts rank.
            bounds, start = [], 0
            for ep in segments:
                bounds.append((start, start + ep["length"], ep))
                start += ep["length"]
            starts = [b[0] for b in bounds]
            rank = {p: i for i, p in enumerate(pts)}

            # One pass over the packets, switching output whenever the episode changes.
            outputs = {}
            first_key = {}
            with av.open(str(src_file)) as inp:
                stream = inp.streams.video[0]
                for packet in inp.demux(stream):
                    if packet.dts is None:
                        continue
                    seg = bisect.bisect_right(starts, rank[packet.pts]) - 1
                    lo, _, ep = bounds[seg]
                    idx = ep["episode_index"]
                    if idx not in outputs:
                        # A copy is only exact if the episode's first packet in decode
                        # order is its first frame and a keyframe.
                        first_key[idx] = packet.is_keyframe and rank[packet.pts] == lo
                        if not first_key[idx]:
                            outputs[idx] = None
                            continue
                        out = dst / V21_VIDEO_PATH.format(episode_chunk=idx // chunks,
                                                          video_key=key, episode_index=idx)
                        out.parent.mkdir(parents=True, exist_ok=True)
                        container = av.open(str(out), "w", options={"movflags": "faststart"})
                        ost = container.add_stream_from_template(template=stream, opaque=True)
                        ost.time_base = stream.time_base
                        outputs[idx] = (container, ost, pts[lo])
                    if outputs[idx] is None:
                        continue
                    container, ost, offset = outputs[idx]
                    packet.pts -= offset
                    packet.dts -= offset
                    packet.stream = ost
                    container.mux(packet)
            for idx, value in outputs.items():
                if value is not None:
                    value[0].close()
            for lo, hi, ep in bounds:
                idx = ep["episode_index"]
                if not first_key.get(idx):
                    out = dst / V21_VIDEO_PATH.format(episode_chunk=idx // chunks,
                                                      video_key=key, episode_index=idx)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    _reencode(src_file, out, lo, hi - lo, fps)
                    report.append(f"{key} episode {idx}: re-encoded (did not start on a keyframe)")
    return report


# --------------------------------------------------------------------------- verify
def _frame_hashes(path: Path):
    """(seconds, sha1 of pixels) for every decoded frame, in presentation order."""
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            yield float(frame.pts * stream.time_base), hashlib.sha1(
                frame.to_ndarray(format="rgb24").tobytes()).hexdigest()


def verify(src: Path, dst: Path, episodes: list[dict], video_keys: list[str], chunks: int,
           fps: int, decode: bool, copied: set) -> list[str]:
    notes = []
    # Data: every row equal to the source's.
    by_file = defaultdict(list)
    for ep in episodes:
        by_file[(ep["data/chunk_index"], ep["data/file_index"])].append(ep)
    for (chunk, file), eps in by_file.items():
        table = pq.read_table(src / V3_DATA_PATH.format(chunk_index=chunk, file_index=file))
        for ep in eps:
            want = table.filter(pc.equal(table["episode_index"], ep["episode_index"]))
            got = pq.read_table(dst / V21_DATA_PATH.format(
                episode_chunk=ep["episode_index"] // chunks, episode_index=ep["episode_index"]))
            if not got.equals(want):  # values and types; schema metadata is compared below
                raise SystemExit(f"episode {ep['episode_index']}: data differs from the source")
            if b'"List"' in (got.schema.metadata or {}).get(b"huggingface", b""):
                raise SystemExit(f"episode {ep['episode_index']}: datasets 4 feature types remain")
            ts = np.asarray(got["timestamp"].to_pylist(), dtype=np.float64)
            fi = np.asarray(got["frame_index"].to_pylist())
            if not np.allclose(ts, fi / fps, atol=1e-4):
                raise SystemExit(f"episode {ep['episode_index']}: timestamp != frame_index / fps")
    notes.append(f"data: {len(episodes)} episodes equal to the source, row for row")
    if not decode:
        return notes
    # Video: decode both sides; every frame identical and at frame_index / fps.
    for key in video_keys:
        by_file = defaultdict(list)
        for ep in episodes:
            by_file[(ep[f"videos/{key}/chunk_index"], ep[f"videos/{key}/file_index"])].append(ep)
        n_frames = 0
        for (chunk, file), eps in sorted(by_file.items()):
            src_file = src / V3_VIDEO_PATH.format(video_key=key, chunk_index=chunk, file_index=file)
            source = [h for _, h in _frame_hashes(src_file)]
            start = 0
            for ep in _segments(eps, key):
                idx = ep["episode_index"]
                out = dst / V21_VIDEO_PATH.format(episode_chunk=idx // chunks, video_key=key,
                                                  episode_index=idx)
                frames = list(_frame_hashes(out))
                if len(frames) != ep["length"]:
                    raise SystemExit(f"{out}: {len(frames)} frames, expected {ep['length']}")
                for i, (t, h) in enumerate(frames):
                    if abs(t - i / fps) > 1e-4:
                        raise SystemExit(f"{out}: frame {i} at {t:.6f} s, expected {i / fps:.6f} s")
                    if (key, idx) in copied and h != source[start + i]:
                        raise SystemExit(f"{out}: frame {i} differs from the source frame")
                start += ep["length"]
                n_frames += len(frames)
        notes.append(f"{key}: {n_frames} frames decoded, at frame_index / fps, "
                     "pixel-identical to the source where copied")
    return notes


# --------------------------------------------------------------------------- main
def convert(src: Path, dst: Path, check: bool = True, decode: bool = True) -> list[str]:
    info = json.loads((src / "meta" / "info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        raise SystemExit(f"{src} is {info.get('codebase_version')}, not v3.0")
    features = info["features"]
    video_keys = [k for k, f in features.items() if f["dtype"] == "video"]
    if any(f["dtype"] == "image" for f in features.values()):
        raise SystemExit("image (PNG) features are not supported; only video and numeric")
    episodes = load_episodes(src)
    tasks = load_tasks(src)
    chunks, fps = info.get("chunks_size", 1000), info["fps"]

    (dst / "meta").mkdir(parents=True)
    (dst / "meta" / "info.json").write_text(
        json.dumps(convert_info(info, len(episodes), video_keys), indent=4))
    with (dst / "meta" / "tasks.jsonl").open("w") as f:
        for index, task in tasks:
            f.write(json.dumps({"task_index": index, "task": task}) + "\n")
    with (dst / "meta" / "episodes.jsonl").open("w") as f, \
            (dst / "meta" / "episodes_stats.jsonl").open("w") as g:
        for ep in episodes:
            f.write(json.dumps({"episode_index": ep["episode_index"], "tasks": list(ep["tasks"]),
                                "length": ep["length"]}) + "\n")
            g.write(json.dumps({"episode_index": ep["episode_index"],
                                "stats": episode_stats(ep, features)}) + "\n")
    if (src / "meta" / "stats.json").exists():
        shutil.copy2(src / "meta" / "stats.json", dst / "meta" / "stats.json")
    for extra in sorted((src / "meta").iterdir()):
        if extra.name in ("info.json", "stats.json", "tasks.parquet", "episodes"):
            continue
        (shutil.copytree if extra.is_dir() else shutil.copy2)(extra, dst / "meta" / extra.name)
    for extra in sorted(src.iterdir()):
        if extra.name in ("meta", "data", "videos"):
            continue
        (shutil.copytree if extra.is_dir() else shutil.copy2)(extra, dst / extra.name)

    frames = convert_data(src, dst, episodes, chunks)
    report = convert_videos(src, dst, episodes, video_keys, chunks, fps)
    reencoded = {(line.split(" episode ")[0], int(line.split(" episode ")[1].split(":")[0]))
                 for line in report}
    copied = {(k, e["episode_index"]) for k in video_keys for e in episodes} - reencoded
    report.insert(0, f"{len(episodes)} episodes, {frames} frames, {len(tasks)} tasks, "
                     f"cameras {video_keys}")
    if check:
        report += verify(src, dst, episodes, video_keys, chunks, fps, decode, copied)
    return report


def main():
    args = parse_args()
    src, dst = args.src.expanduser().resolve(), args.dst.expanduser().resolve()
    if dst.exists():
        if not args.force:
            raise SystemExit(f"{dst} exists (use --force to replace)")
        shutil.rmtree(dst)
    try:
        report = convert(src, dst, check=True, decode=not args.no_verify)
    except BaseException:
        # A half-written v2.1 tree must not be mistaken for a dataset.
        shutil.rmtree(dst, ignore_errors=True)
        raise
    print(f"{src} (v3.0) -> {dst} (v2.1)")
    for line in report:
        print("  " + line)


if __name__ == "__main__":
    main()
