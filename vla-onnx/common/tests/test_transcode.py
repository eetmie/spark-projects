"""vla_common.dataset.transcode on a synthetic AV1 dataset."""

import json
import subprocess
import sys

import av
import numpy as np
import pytest

from vla_common.dataset.transcode import transcode

H, W, FPS, N = 64, 96, 30, 20
KEY = "observation.images.front"


def make_av1(path, n):
    path.parent.mkdir(parents=True)
    rng = np.random.default_rng(0)
    with av.open(str(path), "w") as c:
        s = c.add_stream("libsvtav1", rate=FPS)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        base = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        for i in range(n):
            f = av.VideoFrame.from_ndarray(np.roll(base, i, axis=1), format="rgb24")
            f.pts = i
            for p in s.encode(f):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


@pytest.fixture
def av1_dataset(tmp_path):
    src = tmp_path / "src"
    (src / "meta").mkdir(parents=True)
    (src / "data").mkdir()
    (src / "data/x.parquet").write_bytes(b"data")
    info = {"fps": FPS, "features": {KEY: {"dtype": "video", "shape": [H, W, 3],
                                           "info": {"video.codec": "av1", "video.pix_fmt": "yuv420p"}}}}
    (src / "meta/info.json").write_text(json.dumps(info))
    make_av1(src / f"videos/{KEY}/chunk-000/file-000.mp4", N)
    return src


def test_transcode(av1_dataset, tmp_path):
    dst = tmp_path / "dst"
    reports = transcode(av1_dataset, dst, jobs=1)
    assert [r["frames"] for r in reports] == [N]
    assert reports[0]["psnr_db"] > 30

    info = json.loads((dst / "meta/info.json").read_text())
    assert info["features"][KEY]["info"] == {"video.codec": "h264", "video.pix_fmt": "yuv420p", "video.g": 2, "video.crf": 18}
    assert (dst / "data/x.parquet").read_bytes() == b"data"
    assert json.loads((av1_dataset / "meta/info.json").read_text())["features"][KEY]["info"]["video.codec"] == "av1"

    out = dst / f"videos/{KEY}/chunk-000/file-000.mp4"
    with av.open(str(out)) as c:
        assert c.streams.video[0].codec_context.name == "h264"
    # The decoder training uses: every frame lands at its timestamp.
    code = ("import sys; from lerobot.datasets.video_utils import decode_video_frames_torchcodec as d;"
            f"f = d(sys.argv[1], [i / {FPS} for i in range({N})], 1e-4); assert f.shape[0] == {N}, f.shape")
    subprocess.run([sys.executable, "-c", code, str(out)], check=True)

    with pytest.raises(SystemExit, match="exists"):
        transcode(av1_dataset, dst)
