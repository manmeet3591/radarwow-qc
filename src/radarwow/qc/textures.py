"""Range-texture features used by clutter and non-meteorological echo detection.

All features are computed along range (axis 1) on (azimuth, range) arrays with NaN-aware
moving windows, so gaps and censored gates do not poison neighbours.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


def _moving_mean(x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(x)
    s = uniform_filter1d(np.where(valid, x, 0.0), n, axis=1, mode="constant")
    c = uniform_filter1d(valid.astype(float), n, axis=1, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / c, np.nan), c * n


def tdbz(z: np.ndarray, n: int = 9) -> np.ndarray:
    """Reflectivity texture TDBZ: mean squared gate-to-gate dBZ difference over n gates (dB^2)."""
    d2 = np.full(z.shape, np.nan)
    d2[:, 1:] = np.diff(z, axis=1) ** 2
    m, cnt = _moving_mean(d2, n)
    return np.where(cnt >= n // 2, m, np.nan)


def spin(z: np.ndarray, n: int = 11, threshold_db: float = 5.0) -> np.ndarray:
    """SPIN: percentage of gradient sign changes, counting only steps with |dZ| > threshold."""
    dz = np.diff(z, axis=1)
    big = np.abs(dz) > threshold_db
    s = np.sign(dz)
    change = np.zeros(z.shape)
    flip = (s[:, 1:] * s[:, :-1] < 0) & big[:, 1:] & big[:, :-1]
    change[:, 2:] = flip
    valid = np.isfinite(z).astype(float)
    num = uniform_filter1d(change, n, axis=1, mode="constant")
    den = uniform_filter1d(valid, n, axis=1, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0.5, 100.0 * num / den, np.nan)


def sd(x: np.ndarray, n: int = 7) -> np.ndarray:
    """Moving standard deviation along range."""
    m, cnt = _moving_mean(x, n)
    m2, _ = _moving_mean(x * x, n)
    return np.where(cnt >= 3, np.sqrt(np.maximum(m2 - m * m, 0.0)), np.nan)


def sd_phase(phi_deg: np.ndarray, n: int = 7) -> np.ndarray:
    """Circular standard deviation of a phase field (deg) along range: robust to 0/360 wrapping."""
    a = np.deg2rad(phi_deg)
    c, cnt = _moving_mean(np.cos(a), n)
    s, _ = _moving_mean(np.sin(a), n)
    r = np.clip(np.hypot(c, s), 1e-12, 1.0)
    return np.where(cnt >= 3, np.rad2deg(np.sqrt(-2.0 * np.log(r))), np.nan)


def ramp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Fuzzy membership: 0 below lo, 1 above hi, linear between (NaN stays NaN)."""
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)
