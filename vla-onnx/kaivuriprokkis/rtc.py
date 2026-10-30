"""Chunk-to-chunk consistency (RTC) for pi0.5: the NumPy reference ("oracle").

Pure NumPy, no torch. The exporter uses it to write RTC reference fixtures, and the
kaivuriprokkis runtime's device op and host library must match it
(kaivuriprokkis native_stack/notes/RTC_PLAN.md, P1/P2).

Mode A (inference-time inpainting) follows the BEHAVIOR-1K / LeHome code
(IliaLarchenko/lehome_solution, models/pi_modified.py, `sample_actions` step loop and
`_precompute_all_correction_matrices`). After each Euler step, with t = the new time:

    if t > t_ip:
        x_O   <- (1 - t) * x0_O + t * z_O                  hard set (stays on the flow path)
        corr  =  ((x_desired - x_new) masked to O) @ C      C[o, u] = (Sigma_UO Sigma_OO^-1)[u, o]
        x_U   += corr  unless max|corr| > limit             stability guard skips only the push

Sigma is the *correlation* matrix of flattened normalized action chunks with shrinkage
`beta * Sigma + (1 - beta) * I` (reference default beta 0.5).

Mode C (training-time clean prefix): rows < d are set to the clean anchor at the start and
after every step, and are fed the t = 0 style. No correction. The anchor covers the full
padded width: the D robot dims, then zeros. That is what training sees on those rows
(OpenPI's PadStatesAndActions zero-pads the normalized actions, and a t = 0 row is the clean
action itself). Leaving the padding on the noise path would feed the model prefix rows it
never saw in training.

Layout: chunks are [H, W] with W the padded model width (32); only the first D robot
dims are anchored or corrected. Flattened index = row * D + dim.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODES = ("off", "inpaint", "clean")


def openpi_times(steps: int) -> np.ndarray:
    """[steps + 1] float32: t before each step, then the final time, accumulated exactly as
    OpenPI's PyTorch loop does (`time = 1.0; time += dt` in float32)."""
    dt = np.float32(-1.0 / steps)
    times, t = [], np.float32(1.0)
    for _ in range(steps + 1):
        times.append(t)
        t = np.float32(t + dt)
    return np.asarray(times, np.float32)


@dataclass(frozen=True)
class Schedule:
    """Per post-step blend coefficients. Entry s applies after Euler step s, at time t[s+1]."""
    mode: str
    alpha: np.ndarray  # [steps] float32, weight of x0
    beta: np.ndarray   # [steps] float32, weight of z
    gate: np.ndarray   # [steps] bool
    times: np.ndarray  # [steps + 1] float32

    @property
    def correct(self) -> bool:
        return self.mode == "inpaint"


def schedule(mode: str, steps: int, time_threshold: float = 0.3) -> Schedule:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    times = openpi_times(steps)
    new = times[1:]
    if mode == "inpaint":
        alpha, beta = (np.float32(1) - new).astype(np.float32), new.copy()
        gate = new > np.float32(time_threshold)
    elif mode == "clean":
        alpha, beta = np.ones(steps, np.float32), np.zeros(steps, np.float32)
        gate = np.ones(steps, bool)
    else:
        alpha, beta, gate = np.zeros(steps, np.float32), np.zeros(steps, np.float32), np.zeros(steps, bool)
    return Schedule(mode, alpha, beta, gate, times)


# ── statistics ────────────────────────────────────────────────────────────────

def chunk_moments(chunks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean [H*D] and covariance [H*D, H*D] (float64, ddof=1) of [N, H, D] chunks."""
    flat = np.asarray(chunks, np.float64).reshape(len(chunks), -1)
    if len(flat) < 2:
        raise ValueError("need at least two chunks")
    return flat.mean(0), np.cov(flat, rowvar=False)


def correlation(cov: np.ndarray) -> np.ndarray:
    """Reference correlation: constant dims (var < 1e-10) get a zero row/column, unit diagonal."""
    cov = np.asarray(cov, np.float64)
    diag = np.diag(cov).copy()
    constant = diag < 1e-10
    safe = np.where(constant, 1.0, diag)
    corr = cov / np.sqrt(safe[:, None] * safe[None, :])
    corr[constant, :] = 0.0
    corr[:, constant] = 0.0
    np.fill_diagonal(corr, 1.0)
    return corr


def reference_sigma(cov: np.ndarray, beta: float = 0.5) -> np.ndarray:
    """The Sigma the reference propagates with: L_reg @ L_reg.T, where the stats store
    chol(corr + 1e-6 I) and loading shrinks it to beta * Sigma + (1 - beta) * I."""
    corr = correlation(cov)
    n = len(corr)
    sigma = corr + 1e-6 * np.eye(n)
    if np.linalg.eigvalsh(sigma).min() <= 0:  # reference: stronger jitter when not PD
        sigma = corr + max(1e-5, -np.linalg.eigvalsh(corr).min() + 1e-5) * np.eye(n)
    return beta * sigma + (1.0 - beta) * np.eye(n)


def correction_matrix(sigma: np.ndarray, rows: int, dims: int, horizon: int) -> np.ndarray:
    """Full-space [H*D, H*D] float32 C with C[o, u] = (Sigma_UO Sigma_OO^-1)[u, o] for the
    first `rows` rows observed; zero elsewhere, so `delta @ C` lands on free positions."""
    flat = horizon * dims
    if sigma.shape != (flat, flat):
        raise ValueError(f"sigma is {sigma.shape}, expected {(flat, flat)}")
    if not 0 <= rows <= horizon:
        raise ValueError("rows outside the horizon")
    full = np.zeros((flat, flat), np.float32)
    if rows == 0:
        return full
    o = np.arange(rows * dims)
    u = np.arange(rows * dims, flat)
    s_oo = sigma[np.ix_(o, o)]
    s_uo = sigma[np.ix_(u, o)]
    jitter = 1e-6 * max(float(np.mean(np.diag(s_oo))), 1.0)
    small = np.linalg.solve(s_oo + jitter * np.eye(len(o)), s_uo.T).T  # [|U|, |O|]
    full[np.ix_(o, u)] = small.T.astype(np.float32)
    return full


class Corrections:
    """C for every anchor length 0..max_rows, built once. A longer request is an error:
    the reference repo silently clipped an out-of-range index (RTC_SCOPE item 4)."""

    def __init__(self, sigma: np.ndarray, dims: int, horizon: int, max_rows: int):
        self.dims, self.horizon, self.max_rows = dims, horizon, max_rows
        self._c = [correction_matrix(sigma, d, dims, horizon) for d in range(max_rows + 1)]

    def __call__(self, rows: int) -> np.ndarray:
        if not 0 <= rows <= self.max_rows:
            raise ValueError(f"anchor of {rows} rows exceeds the cached {self.max_rows}")
        return self._c[rows]


# ── chunk bookkeeping ─────────────────────────────────────────────────────────

def elapsed_rows(new_observation_ns: int, old_observation_ns: int, fps: float) -> int:
    """Rows of the old chunk already played when the new observation was taken."""
    if new_observation_ns < old_observation_ns:
        raise ValueError("new observation is older than the previous chunk's")
    frame_ns = 1e9 / fps
    return int(np.ceil((new_observation_ns - old_observation_ns) / frame_ns - 1e-9))


def anchor(previous: np.ndarray, elapsed: int, rows: int, dims: int) -> np.ndarray | None:
    """Old chunk rows [elapsed, elapsed + rows) become new rows [0, rows); None if the old
    chunk is too short (the caller then runs that chunk with RTC off)."""
    if rows == 0 or elapsed < 0 or elapsed + rows > len(previous):
        return None
    return np.ascontiguousarray(previous[elapsed:elapsed + rows, :dims], np.float32)


# ── the per-step operation ────────────────────────────────────────────────────

def mask(rows: int, dims: int, horizon: int) -> np.ndarray:
    m = np.zeros((horizon, dims), np.float32)
    m[:rows] = 1
    return m.reshape(-1)


def initial(z: np.ndarray, x0: np.ndarray | None, sched: Schedule) -> np.ndarray:
    """Starting state: the noise, with the prefix set to the clean anchor in Mode C."""
    x = np.array(z, np.float32, copy=True)
    if x0 is not None and sched.mode == "clean":
        x[:len(x0), :x0.shape[1]] = x0
        x[:len(x0), x0.shape[1]:] = 0  # zero padding, as trained
    return x


def blend(x_new: np.ndarray, z: np.ndarray, x0: np.ndarray | None, sched: Schedule, step: int,
          c: np.ndarray | None = None, limit: float = 1.0) -> tuple[np.ndarray, dict]:
    """Apply the post-step blend to x_new [H, W] after Euler step `step`. Returns the new
    state and what happened ({gated, corrected, max_correction})."""
    info = {"gated": False, "corrected": False, "max_correction": 0.0}
    if x0 is None or sched.mode == "off" or not sched.gate[step]:
        return x_new, info
    rows, dims = x0.shape
    horizon = x_new.shape[0]
    info["gated"] = True
    out = np.array(x_new, np.float32, copy=True)
    robot = out[:, :dims].reshape(-1)
    m = mask(rows, dims, horizon)
    targets = np.zeros(horizon * dims, np.float32)
    targets[:rows * dims] = x0.reshape(-1)
    zf = np.ascontiguousarray(z[:, :dims], np.float32).reshape(-1) * m
    desired = (sched.alpha[step] * targets + sched.beta[step] * zf).astype(np.float32)
    delta = (desired - robot) * m
    robot = np.where(m > 0.5, desired, robot).astype(np.float32)
    if sched.correct:
        if c is None:
            raise ValueError("inpaint mode needs a correction matrix")
        corr = (delta @ c).astype(np.float32) * (np.float32(1) - m)
        peak = float(np.max(np.abs(corr)))
        info["max_correction"] = peak
        if peak <= limit:
            robot = robot + corr
            info["corrected"] = True
    out[:, :dims] = robot.reshape(horizon, dims)
    if sched.mode == "clean":
        out[:rows, dims:] = 0  # the prefix's padding stays zero, as trained
    return out, info


def denoise(velocity, z: np.ndarray, sched: Schedule, x0: np.ndarray | None = None,
            c: np.ndarray | None = None, limit: float = 1.0) -> tuple[np.ndarray, list[dict]]:
    """Reference Euler loop. `velocity(step, x)` returns v [H, W] float32."""
    steps = len(sched.alpha)
    dt = np.float32(-1.0 / steps)
    x = initial(z, x0, sched)
    log = []
    for s in range(steps):
        x = (x + dt * np.asarray(velocity(s, x), np.float32)).astype(np.float32)
        x, info = blend(x, z, x0, sched, s, c, limit)
        log.append(info)
    return x, log


# ── metrics ───────────────────────────────────────────────────────────────────

def join_metrics(new: np.ndarray, previous: np.ndarray, elapsed: int, rows: int, dims: int,
                 switch_row: int | None = None) -> dict:
    """How well the new chunk continues the old one (normalized units, robot dims).

    overlap_max_abs: max |new[0:rows) - old[elapsed:elapsed+rows)| (rows > 0)
    switch_jump:     |new[k] - old[elapsed+k]| at k = switch_row (default: the first free
                     row), the jump when execution switches to the new plan there; the
                     runtime switches at about latency x fps rows after the observation
    first_free_step: |new[rows] - new[rows-1]| against median_step, the median row-to-row
                     change of the old chunk
    """
    new, previous = np.asarray(new)[:, :dims], np.asarray(previous)[:, :dims]
    k = min(rows if switch_row is None else switch_row, len(previous) - elapsed - 1)
    out = {"median_step": float(np.median(np.abs(np.diff(previous, axis=0)).max(-1))), "switch_row": int(k)}
    out["switch_jump"] = float(np.abs(new[k] - previous[elapsed + k]).max())
    if rows > 0:
        out["overlap_max_abs"] = float(np.abs(new[:rows] - previous[elapsed:elapsed + rows]).max())
        out["first_free_step"] = float(np.abs(new[rows] - new[rows - 1]).max())
    return out
