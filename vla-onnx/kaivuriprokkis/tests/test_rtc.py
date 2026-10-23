"""CPU tests for the RTC oracle (rtc.py). Run: python -m unittest discover -s tests"""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import rtc  # noqa: E402

H, D, W = 12, 3, 8


def spd(n, rng):
    a = rng.standard_normal((n, n))
    return a @ a.T + 0.1 * np.eye(n)


def linear_velocity(rate):
    # Row-local velocity, so free rows do not depend on the prefix.
    return lambda step, x: (rate * x).astype(np.float32)


class Schedule(unittest.TestCase):
    def test_openpi_time_grid(self):
        t = rtc.openpi_times(10)
        self.assertEqual(t.dtype, np.float32)
        self.assertEqual(t[0], 1.0)
        self.assertEqual(len(t), 11)
        self.assertLess(abs(float(t[-1])), 1e-6)
        manifest = HERE.parents[1] / "pi05/runs/bundle-smoke9/conditioning_manifest.json"
        if manifest.exists():  # the exporter's conditioning grid (prepare_conditioning.py)
            bits = json.loads(manifest.read_text())["timesteps_fp32_bits"]
            self.assertEqual(t[:10].view(np.uint32).tolist(), bits)

    def test_inpaint_coefficients(self):
        s = rtc.schedule("inpaint", 10, 0.4)
        np.testing.assert_array_equal(s.beta, s.times[1:])
        np.testing.assert_array_equal(s.alpha, np.float32(1) - s.times[1:])
        self.assertEqual(s.gate.tolist(), [bool(t > np.float32(0.4)) for t in s.times[1:]])
        self.assertEqual(int(s.gate.sum()), 5)  # 0.9 .. 0.5
        self.assertEqual(int(rtc.schedule("inpaint", 10, 0.5).gate.sum()), 4)

    def test_clean_and_off(self):
        self.assertTrue(rtc.schedule("clean", 10).gate.all())
        self.assertFalse(rtc.schedule("off", 10).gate.any())
        with self.assertRaises(ValueError):
            rtc.schedule("guided", 10)


class Correction(unittest.TestCase):
    def test_conditional_mean(self):
        rng = np.random.default_rng(0)
        sigma = spd(H * D, rng)
        for rows in (1, 4, H - 1):
            c = rtc.correction_matrix(sigma, rows, D, H)
            o, u = np.arange(rows * D), np.arange(rows * D, H * D)
            delta = np.zeros(H * D)
            delta[o] = rng.standard_normal(len(o))
            expected = sigma[np.ix_(u, o)] @ np.linalg.solve(sigma[np.ix_(o, o)], delta[o])
            got = delta.astype(np.float32) @ c
            np.testing.assert_allclose(got[u], expected, rtol=1e-3, atol=1e-4)
            np.testing.assert_array_equal(got[o], 0)
            # Index convention of the reference: C[o, u] maps observed o to free u.
            self.assertTrue(np.all(c[np.ix_(u, u)] == 0) and np.all(c[:, o] == 0))

    def test_cache_refuses_longer_anchor(self):
        cache = rtc.Corrections(spd(H * D, np.random.default_rng(1)), D, H, 4)
        self.assertEqual(cache(4).shape, (H * D, H * D))
        with self.assertRaises(ValueError):
            cache(5)

    def test_correlation(self):
        rng = np.random.default_rng(2)
        cov = spd(6, rng)
        cov[2, :] = cov[:, 2] = 0  # constant dim
        corr = rtc.correlation(cov)
        np.testing.assert_allclose(np.diag(corr), 1)
        self.assertTrue(np.all(corr[2, [0, 1, 3, 4, 5]] == 0))
        sigma = rtc.reference_sigma(cov, 0.5)
        np.linalg.cholesky(sigma)
        np.testing.assert_allclose(np.diag(sigma), 1 + 0.5e-6)


class Blend(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.z = rng.standard_normal((H, W)).astype(np.float32)
        self.x0 = rng.standard_normal((4, D)).astype(np.float32)
        self.c = rtc.correction_matrix(rtc.reference_sigma(spd(H * D, rng)), 4, D, H)

    def test_off_is_bit_exact(self):
        v = linear_velocity(np.float32(-0.7))
        plain, _ = rtc.denoise(v, self.z, rtc.schedule("off", 10))
        for sched, x0 in ((rtc.schedule("inpaint", 10), None), (rtc.schedule("off", 10), self.x0)):
            got, _ = rtc.denoise(v, self.z, sched, x0, self.c)
            self.assertTrue(np.array_equal(got, plain))
        reference = self.z.copy()
        for _ in range(10):
            reference = (reference + np.float32(-0.1) * (np.float32(-0.7) * reference)).astype(np.float32)
        self.assertTrue(np.array_equal(plain, reference))

    def test_inpaint_stays_on_the_flow_path(self):
        sched = rtc.schedule("inpaint", 10, 0.0)
        x, log = rtc.denoise(linear_velocity(np.float32(-0.3)), self.z, sched, self.x0, self.c, limit=np.inf)
        t = sched.times
        # Last gated step is the one whose new time is still > 0: the prefix is exactly on
        # the path there; the final ungated step then moves it by the model only.
        gated = [i for i, e in enumerate(log) if e["gated"]]
        self.assertEqual(gated, [i for i in range(10) if t[i + 1] > 0])
        self.assertTrue(all(e["corrected"] for e in log if e["gated"]))
        # Padded dims are never touched.
        plain, _ = rtc.denoise(linear_velocity(np.float32(-0.3)), self.z, rtc.schedule("off", 10))
        np.testing.assert_array_equal(x[:, D:], plain[:, D:])

    def test_hard_set_value(self):
        sched = rtc.schedule("inpaint", 10, 0.3)
        x_new = np.random.default_rng(4).standard_normal((H, W)).astype(np.float32)
        out, info = rtc.blend(x_new, self.z, self.x0, sched, 0, self.c, limit=np.inf)
        t = sched.times[1]
        np.testing.assert_array_equal(out[:4, :D], (np.float32(1) - t) * self.x0 + t * self.z[:4, :D])
        np.testing.assert_array_equal(out[4:, D:], x_new[4:, D:])
        self.assertTrue(info["corrected"])

    def test_stability_guard_skips_only_the_push(self):
        sched = rtc.schedule("inpaint", 10, 0.3)
        x_new = (100 * np.random.default_rng(5).standard_normal((H, W))).astype(np.float32)
        out, info = rtc.blend(x_new, self.z, self.x0, sched, 0, self.c, limit=1.0)
        self.assertFalse(info["corrected"])
        self.assertGreater(info["max_correction"], 1.0)
        np.testing.assert_array_equal(out[4:], x_new[4:])
        t = sched.times[1]
        np.testing.assert_array_equal(out[:4, :D], (np.float32(1) - t) * self.x0 + t * self.z[:4, :D])

    def test_below_threshold_is_untouched(self):
        sched = rtc.schedule("inpaint", 10, 0.5)
        x_new = np.ones((H, W), np.float32)
        out, info = rtc.blend(x_new, self.z, self.x0, sched, 6, self.c)  # t = 0.3
        self.assertFalse(info["gated"])
        self.assertTrue(np.array_equal(out, x_new))

    def test_clean_prefix_equals_anchor_every_step(self):
        sched = rtc.schedule("clean", 10)
        v = linear_velocity(np.float32(-0.5))
        x = rtc.initial(self.z, self.x0, sched)
        np.testing.assert_array_equal(x[:4, :D], self.x0)
        for s in range(10):
            x = (x + np.float32(-0.1) * v(s, x)).astype(np.float32)
            x, _ = rtc.blend(x, self.z, self.x0, sched, s)
            np.testing.assert_array_equal(x[:4, :D], self.x0)
        plain, _ = rtc.denoise(v, self.z, rtc.schedule("off", 10))
        np.testing.assert_array_equal(x[4:], plain[4:])


class Anchor(unittest.TestCase):
    def test_alignment(self):
        previous = np.arange(50 * W, dtype=np.float32).reshape(50, W)
        self.assertEqual(rtc.elapsed_rows(1_000_000_000 + 433_000_000, 1_000_000_000, 30), 13)
        self.assertEqual(rtc.elapsed_rows(1_000_000_000 + 2 * 33_333_333, 1_000_000_000, 30), 2)
        x0 = rtc.anchor(previous, 13, 15, 4)
        np.testing.assert_array_equal(x0, previous[13:28, :4])
        self.assertIsNone(rtc.anchor(previous, 40, 15, 4))
        self.assertIsNone(rtc.anchor(previous, 0, 0, 4))

    def test_join_metrics(self):
        previous = np.linspace(0, 1, 50)[:, None].repeat(W, 1).astype(np.float32)
        new = previous[10:].copy()
        new = np.concatenate([new, new[-1:].repeat(10, 0)])
        m = rtc.join_metrics(new, previous, 10, 15, 4)
        self.assertEqual(m["overlap_max_abs"], 0)
        self.assertEqual(m["switch_jump"], 0)


if __name__ == "__main__":
    unittest.main()
