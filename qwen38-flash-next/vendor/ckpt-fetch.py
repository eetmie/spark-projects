#!/usr/bin/env python3
"""A lane's checkpoint, made whole in the cache: the one copy of the fetch that install.sh,
install-image.sh, install-video.sh, install-kolibri.sh and switch-model.sh run, the text lanes
and Kolibri inside their serving image, the image and video lanes in their venv.

Until v1.31 there were seven copies, and they had drifted (found in review, 2026-10-14): only
the Kolibri ones checked that every shard an index names is in the cache before trusting a
snapshot, a switch to the image or video lane gave up at the first network error where every
installer retried, the image switch counted the room in GiB and Kolibri's in GB, and the Xet
exception the flash checkpoint needs lived in the text lanes' copies alone.

    ckpt-fetch.py REPO REV [--allow "PATTERN ..."] [--offline-first] [--need-gb N --free-gb N]
                           [--refs-main pin|fill]

--offline-first asks the cache before the network, and a snapshot every shard of which is there
answers without a single request (a cockpit Load on a box whose cache is complete, offline or
throttled). Without it the Hub is asked, which finds and fetches any missing file. A fetch is
refused before the first byte when the room it needs (--need-gb, less what the cache already
holds of the repo) is over --free-gb. Five attempts, each resuming from the files already
whole, with the byte count of hf-progress.py when it is beside this file. --refs-main says what
becomes of refs/main, which a download by commit never writes and an offline unit that names
the repo resolves: pin points it at the commit fetched (the image and video installers and the
image switch: those units serve the repo by name, so refs/main is what they serve), fill writes
it only where it is missing or holds no commit and never moves one the box serves (the text
lanes, whose units name their revision and whose 1m unit resolves "main" offline, and the
video switch). Kolibri's units name their revision: neither. The snapshot's path is the last
line printed.
"""
import argparse
import contextlib
import fnmatch
import importlib.util
import json
import os
import pathlib
import re
import time

from huggingface_hub import constants, snapshot_download

ATTEMPTS = 5


def reporting(*args, **kwargs):
    """hf-progress.py's byte count while the rest comes; a progress line is never worth a download."""
    try:
        spec = importlib.util.spec_from_file_location(
            "hf_progress", os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf-progress.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.reporting(*args, **kwargs)
    except Exception:
        return contextlib.nullcontext()


def allowed(rel, allow):
    """huggingface_hub's own rule for allow_patterns (utils/_paths.py, filter_repo_objects):
    fnmatchcase on the path relative to the snapshot, a pattern ending in / meaning its content."""
    return not allow or any(fnmatch.fnmatchcase(rel, p + "*" if p.endswith("/") else p) for p in allow)


def complete(path, allow=None):
    """Every shard each index of the snapshot names is there (a file appears in a snapshot only
    once its blob is whole, and a link to a blob that is not counts as missing). huggingface_hub's
    own offline check sees a missing file only when it cached the repo's tree, which a download
    made with another tool or an older hub never wrote, and the engine then dies at every start.
    A snapshot without any index holds its weights in single files: one at least must be there.
    Only what the allow patterns cover counts: a snapshot keeps the index of a part a full pull
    began, as the reference box's MiniMax-H3 keeps Ref2VA's transformer index with none of its 13
    shards, beside the whole FL2VA partition its lane serves (found by the proof on the box,
    2026-10-14). Files no index names (a diffusion model's single-file parts) are not checked
    here: the Hub is, on a fetch."""
    root = pathlib.Path(path)
    indexes = [p for p in root.rglob("*.safetensors.index.json") if allowed(p.relative_to(root).as_posix(), allow)]
    if not indexes:
        return any(p.exists() for p in root.rglob("*.safetensors") if allowed(p.relative_to(root).as_posix(), allow))
    for index in indexes:
        try:
            shards = set(json.loads(index.read_text())["weight_map"].values())
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return False
        if not shards or not all((index.parent / s).exists() for s in shards):
            return False
    return True


def download(repo, rev, kw):
    """snapshot_download, with the one exception to the Xet ban (HF_HUB_DISABLE_XET=1, which every
    caller sets: the Xet backend stalled silently on the reference box, 0-8 MB/s against 89 on
    the classic CDN). The classic CDN refuses any single file over MAX_HTTP_DOWNLOAD_SIZE
    (50,000,000,000 bytes) and says so in a ValueError naming hf_xet. One target ships such a
    file: nvidia/Qwen3.8-Flash-Next-NVFP4 carries its n-gram table as one 53.7 GB safetensors.
    Without this, that target cannot be fetched at all (seen 2026-09-18, 74 GB of the 124 left
    in cache). Xet is turned back on only for the repo the CDN just refused, and off again right
    after, so every other file keeps the transfer path that measured 89 MB/s against its 0-8."""
    try:
        return snapshot_download(repo, revision=rev, **kw)
    except Exception as e:
        if "too large to be downloaded" not in str(e) or not getattr(constants, "HF_HUB_DISABLE_XET", False):
            raise
        print("a file is over the classic CDN limit; retrying this repo with Xet", flush=True)
        constants.HF_HUB_DISABLE_XET = False
        try:
            return snapshot_download(repo, revision=rev, **kw)
        finally:
            constants.HF_HUB_DISABLE_XET = True


def room(repo, need_gb, free_gb):
    """Refused before the first byte rather than stopped by a full disk halfway. What the cache
    already holds of the repo comes off the need: a fetch that stopped resumes from there. In
    bytes: the need is the callers' GB (10^9, as the Hub states repository sizes) and the free
    side df -BG's GiB. The copies before subtracted what the cache held in GB in one (Kolibri's)
    and in GiB in the other (the image switch's), against the same GiB of df."""
    if not free_gb:
        return
    blobs = os.path.join(constants.HF_HUB_CACHE, "models--" + repo.replace("/", "--"), "blobs")
    have = sum(e.stat().st_size for e in os.scandir(blobs) if e.is_file()) if os.path.isdir(blobs) else 0
    rest = max(need_gb * 10**9 - have, 10**9)
    if free_gb * 2**30 < rest:
        raise SystemExit(f"{free_gb} GB free in the cache, and {repo} needs about {-(-rest // 10**9)} GB more: "
                         "free some space first")


def refs_main(path, rev, mode):
    """refs/main, which an offline unit naming the repo resolves and a download by commit never
    writes: pin points it at the commit fetched, fill writes it where it is missing or holds no
    commit and never moves one that does."""
    if not mode:
        return
    sha = os.path.basename(path.rstrip("/"))
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        return
    ref = os.path.join(os.path.dirname(os.path.dirname(path.rstrip("/"))), "refs", "main")
    old = open(ref).read().strip() if os.path.exists(ref) else ""
    if mode == "pin":
        if rev != sha or old == sha:
            return
    elif re.fullmatch(r"[0-9a-f]{40}", old):
        return
    os.makedirs(os.path.dirname(ref), exist_ok=True)
    with open(ref, "w") as f:
        f.write(sha)
    print(f"refs/main -> {sha[:12]}" + (f" (was {old[:12]})" if old else ""), flush=True)


def gb(value):
    """A whole number of GB, and none when the caller's df could not answer (an empty value):
    no room check then, as before."""
    return int(value) if value.strip() else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("repo")
    ap.add_argument("rev")
    ap.add_argument("--allow", default="", help="allow patterns, space separated (the whole repo when empty)")
    ap.add_argument("--offline-first", action="store_true")
    ap.add_argument("--need-gb", type=gb, default=0)
    ap.add_argument("--free-gb", type=gb, default=0)
    ap.add_argument("--refs-main", choices=("pin", "fill"))
    a = ap.parse_args()
    kw = {"allow_patterns": a.allow.split()} if a.allow.split() else {}
    path = None
    if a.offline_first:
        try:
            path = snapshot_download(a.repo, revision=a.rev, local_files_only=True, **kw)
            if not complete(path, kw.get("allow_patterns")):
                path = None
        except Exception:
            path = None
        if path is None:
            print("not complete in the cache, fetching the rest", flush=True)
    if path is None:
        room(a.repo, a.need_gb, a.free_gb)
        with reporting(a.repo, revision=a.rev, **kw):
            for attempt in range(1, ATTEMPTS + 1):  # a new attempt keeps every file already whole
                try:
                    path = download(a.repo, a.rev, kw)
                    break
                except Exception as e:
                    if attempt == ATTEMPTS:
                        raise
                    print(f"download interrupted ({type(e).__name__}), resuming ({attempt}/{ATTEMPTS})...", flush=True)
                    time.sleep(10)
    refs_main(path, a.rev, a.refs_main)
    print(path, flush=True)


if __name__ == "__main__":
    main()
