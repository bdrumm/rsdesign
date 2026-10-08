"""Hardware acceleration for the per-pixel hot loops (colour conversion and colour differences).

On Apple Silicon the GPU is used through MLX (unified memory, no copies to speak of); everywhere else
the same maths runs in NumPy. Measured on an M5 Pro, a 1280x800 CIEDE2000 map takes 342 ms with
scikit-image on the CPU and 14 ms with MLX (25x). The NumPy path matches scikit-image within 0.003 ΔE;
the MLX path uses the GPU's fast float32 transcendentals: mean error ~0.01 ΔE, single pixels up to ~5%
of their value in the CIEDE2000 hue terms. Good for ranking candidates, not for final verdicts.

Use these functions in SEARCH loops (refine trials, icon/font candidate scoring, adversaries). The
independent validator (dt.validate.fidelity) deliberately keeps its own float64 CPU reference so a
final gate verdict never depends on GPU rounding.

    from dt.accel import rgb_to_lab, delta_e76, delta_e2000, backend
    de = delta_e2000(target_rgb, render_rgb)     # float32 (H, W) numpy array

Backends: P['accel.backend'] = 'auto' (mlx when importable, else numpy) | 'mlx' | 'numpy'.
Also: configure_worker_threads() caps OpenCV / BLAS threads inside worker processes so N parallel
workers do not each spawn one thread per core (oversubscription on a shared machine).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from dt.params import P, register

register("accel.backend", "auto", "per-pixel maths backend: auto (mlx on Apple Silicon, else numpy) | mlx | numpy")
register("accel.worker_threads", 2, "OpenCV/BLAS threads per worker process (configure_worker_threads)", (1, 16))

_D65 = (0.95047, 1.0, 1.08883)
_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
_mx = None
_mx_checked = False


def _mlx():
    """The mlx.core module, or None when unavailable/disabled."""
    global _mx, _mx_checked
    if not _mx_checked:
        _mx_checked = True
        try:
            import mlx.core as mx  # noqa: F401
            _mx = mx
        except Exception:
            _mx = None
    return _mx


def backend() -> str:
    want = str(P["accel.backend"]).lower()
    if want == "numpy":
        return "numpy"
    if want in ("auto", "mlx") and _mlx() is not None:
        return "mlx"
    return "numpy"


# ----------------------------------------------------------------------------- shared formulas
def _lab_formula(xp, rgb):
    """sRGB (0..255, float) -> (L, a, b) with array module xp (numpy or mlx.core)."""
    x = rgb / 255.0
    lin = xp.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    m = xp.array(_M) if xp is not np else _M
    xyz = lin @ m.T
    xyz = xyz / (xp.array(np.array(_D65, np.float32)) if xp is not np else np.array(_D65, np.float32))
    f = xp.where(xyz > 0.008856, xp.power(xp.maximum(xyz, 1e-12), 1.0 / 3.0), 7.787 * xyz + 16.0 / 116.0)
    return 116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])


def _de2000_formula(xp, l1, a1, b1, l2, a2, b2):
    pi = np.pi
    c1 = xp.sqrt(a1 * a1 + b1 * b1)
    c2 = xp.sqrt(a2 * a2 + b2 * b2)
    cb = (c1 + c2) / 2.0
    cb7 = cb ** 7
    g = 0.5 * (1.0 - xp.sqrt(cb7 / (cb7 + 25.0 ** 7)))
    a1p, a2p = (1.0 + g) * a1, (1.0 + g) * a2
    c1p, c2p = xp.sqrt(a1p * a1p + b1 * b1), xp.sqrt(a2p * a2p + b2 * b2)
    h1p = xp.arctan2(b1, a1p) % (2 * pi)
    h2p = xp.arctan2(b2, a2p) % (2 * pi)
    dlp, dcp = l2 - l1, c2p - c1p
    zero = (c1p * c2p) == 0
    dh = h2p - h1p
    dh = xp.where(dh > pi, dh - 2 * pi, xp.where(dh < -pi, dh + 2 * pi, dh))
    dh = xp.where(zero, 0.0, dh)
    dhp = 2.0 * xp.sqrt(c1p * c2p) * xp.sin(dh / 2.0)
    lbp, cbp = (l1 + l2) / 2.0, (c1p + c2p) / 2.0
    hs = h1p + h2p
    hbp = xp.where(xp.abs(h1p - h2p) > pi, xp.where(hs < 2 * pi, (hs + 2 * pi) / 2.0, (hs - 2 * pi) / 2.0), hs / 2.0)
    hbp = xp.where(zero, hs, hbp)
    t = (1 - 0.17 * xp.cos(hbp - pi / 6) + 0.24 * xp.cos(2 * hbp) + 0.32 * xp.cos(3 * hbp + pi / 30)
         - 0.20 * xp.cos(4 * hbp - 63 * pi / 180))
    dth = (30 * pi / 180) * xp.exp(-(((hbp * 180 / pi) - 275) / 25) ** 2)
    cbp7 = cbp ** 7
    rc = 2.0 * xp.sqrt(cbp7 / (cbp7 + 25.0 ** 7))
    sl = 1 + 0.015 * (lbp - 50) ** 2 / xp.sqrt(20 + (lbp - 50) ** 2)
    sc, sh = 1 + 0.045 * cbp, 1 + 0.015 * cbp * t
    rt = -xp.sin(2 * dth) * rc
    return xp.sqrt(xp.maximum((dlp / sl) ** 2 + (dcp / sc) ** 2 + (dhp / sh) ** 2 + rt * (dcp / sc) * (dhp / sh), 0.0))


def _as_float(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(rgb)[..., :3], dtype=np.float32)


# ----------------------------------------------------------------------------- public API
def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) sRGB uint8/float -> (H, W, 3) float32 CIE L*a*b* (D65)."""
    x = _as_float(rgb)
    if backend() == "mlx":
        mx = _mlx()
        L, A, B = _lab_formula(mx, mx.array(x))
        out = mx.stack([L, A, B], axis=-1)
        mx.eval(out)
        return np.array(out, dtype=np.float32)
    L, A, B = _lab_formula(np, x)
    return np.stack([L, A, B], axis=-1).astype(np.float32)


def delta_e76(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel CIE76 ΔE (Euclidean Lab distance) as float32 (H, W). Same-size inputs."""
    x, y = _as_float(a), _as_float(b)
    if backend() == "mlx":
        mx = _mlx()
        l1, a1, b1 = _lab_formula(mx, mx.array(x))
        l2, a2, b2 = _lab_formula(mx, mx.array(y))
        d = mx.sqrt((l1 - l2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)
        mx.eval(d)
        return np.array(d, dtype=np.float32)
    l1, a1, b1 = _lab_formula(np, x)
    l2, a2, b2 = _lab_formula(np, y)
    return np.sqrt((l1 - l2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2).astype(np.float32)


def delta_e2000(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel CIEDE2000 ΔE as float32 (H, W). Same-size inputs."""
    x, y = _as_float(a), _as_float(b)
    if backend() == "mlx":
        mx = _mlx()
        d = _de2000_formula(mx, *_lab_formula(mx, mx.array(x)), *_lab_formula(mx, mx.array(y)))
        mx.eval(d)
        return np.array(d, dtype=np.float32)
    return _de2000_formula(np, *_lab_formula(np, x), *_lab_formula(np, y)).astype(np.float32)


def delta_e2000_lab(la: np.ndarray, lb: np.ndarray) -> np.ndarray:
    """CIEDE2000 between two precomputed Lab arrays (..., 3)."""
    if backend() == "mlx":
        mx = _mlx()
        A, B = mx.array(np.asarray(la, np.float32)), mx.array(np.asarray(lb, np.float32))
        d = _de2000_formula(mx, A[..., 0], A[..., 1], A[..., 2], B[..., 0], B[..., 1], B[..., 2])
        mx.eval(d)
        return np.array(d, dtype=np.float32)
    A, B = np.asarray(la, np.float32), np.asarray(lb, np.float32)
    return _de2000_formula(np, A[..., 0], A[..., 1], A[..., 2], B[..., 0], B[..., 1], B[..., 2]).astype(np.float32)


def configure_worker_threads(n: Optional[int] = None) -> int:
    """Cap OpenCV and BLAS threads in THIS process (call at worker start). N workers x one thread per
    core each would oversubscribe the machine; the pool size already provides the parallelism.
    Note: macOS OpenCV builds schedule on Grand Central Dispatch and ignore setNumThreads; there the
    effective control is the number of worker processes (dt.service pool size)."""
    n = int(P["accel.worker_threads"] if n is None else n)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(n)
    try:
        import cv2
        cv2.setNumThreads(n)
    except Exception:
        pass
    return n


# ----------------------------------------------------------------------------- search mode
import contextlib as _contextlib
import threading as _threading

_search = _threading.local()


def search_active() -> bool:
    """True inside ``with accel.search():`` (and only when a GPU backend is active)."""
    return bool(getattr(_search, "depth", 0)) and backend() == "mlx"


@_contextlib.contextmanager
def search():
    """Mark a block as a SEARCH loop (refine trials, candidate scoring): per-pixel maps inside it may use
    the GPU path. Final measurements (bench, dt.validate) run outside it and keep the CPU reference."""
    _search.depth = getattr(_search, "depth", 0) + 1
    try:
        yield
    finally:
        _search.depth -= 1
