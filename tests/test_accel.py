"""dt.accel: GPU (MLX) and NumPy backends agree with the scikit-image references and with each other."""
import numpy as np
import pytest
from skimage import color as skcolor

from dt import accel
from dt.common.image import load_rgb
from dt.params import P

ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))


def _pair():
    a = load_rgb(f"{ROOT}/fixtures/corpus/mwc/mwc_1_000.png")[:300, :400]
    rng = np.random.default_rng(1)
    b = np.clip(a.astype(np.int16) + rng.integers(-12, 13, a.shape), 0, 255).astype(np.uint8)
    b[50:120, 60:200] = (103, 80, 164)  # a real colour difference
    return a, b


# Tolerances per backend: NumPy (float32) matches scikit-image (float64) almost exactly; MLX uses the GPU's
# fast float32 transcendentals, so CIEDE2000 hue terms drift by up to ~5% on single pixels (mean ~0.01 ΔE).
# That is fine for ranking candidates in search loops; final gate verdicts use dt.validate's CPU reference.
TOL = {"numpy": {"lab": 0.01, "de76": 0.01, "de00_max": 0.01, "de00_mean": 0.002},
       "mlx": {"lab": 0.1, "de76": 0.1, "de00_max": 0.5, "de00_mean": 0.02}}


@pytest.mark.parametrize("be", ["numpy", "mlx"])
def test_backends_match_skimage(be):
    if be == "mlx" and accel._mlx() is None:
        pytest.skip("mlx not installed")
    a, b = _pair()
    t = TOL[be]
    old = P["accel.backend"]
    P.set("accel.backend", be)
    try:
        assert accel.backend() == be
        la, lb = skcolor.rgb2lab(a / 255.0), skcolor.rgb2lab(b / 255.0)
        assert np.abs(accel.rgb_to_lab(a) - la).max() < t["lab"]
        assert np.abs(accel.delta_e76(a, b) - np.sqrt(((la - lb) ** 2).sum(-1))).max() < t["de76"]
        ref00 = skcolor.deltaE_ciede2000(la, lb)
        d00 = accel.delta_e2000(a, b)
        assert d00.dtype == np.float32 and d00.shape == a.shape[:2]
        err = np.abs(d00 - ref00)
        assert err.max() < t["de00_max"] and err.mean() < t["de00_mean"], (err.max(), err.mean())
        err2 = np.abs(accel.delta_e2000_lab(accel.rgb_to_lab(a), accel.rgb_to_lab(b)) - ref00)
        assert err2.max() < t["de00_max"] and err2.mean() < t["de00_mean"]
    finally:
        P.set("accel.backend", old)


def test_identical_images_are_zero():
    a, _ = _pair()
    assert float(accel.delta_e2000(a, a).max()) < 1e-3 and float(accel.delta_e76(a, a).max()) < 1e-3


def test_worker_threads_cap(monkeypatch):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.delenv(v, raising=False)
    import cv2
    before = cv2.getNumThreads()
    try:
        assert accel.configure_worker_threads(2) == 2
        assert all(__import__("os").environ[v] == "2" for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"))
        # macOS OpenCV builds schedule on Grand Central Dispatch and ignore setNumThreads; elsewhere it applies
        if cv2.getNumThreads() != before:
            assert cv2.getNumThreads() == 2
    finally:
        cv2.setNumThreads(before)


def test_search_mode_switches_diff_map_only_inside():
    from dt.compare.pixel import diff_map
    a, b = _pair()
    cpu = diff_map(a, b)
    with accel.search():
        fast = diff_map(a, b)
        assert accel.search_active() == (accel.backend() == "mlx")
    assert not accel.search_active()
    assert np.array_equal(diff_map(a, b), cpu)  # outside search: the CPU reference, bit-exact
    assert np.abs(fast - cpu).max() < 0.1
