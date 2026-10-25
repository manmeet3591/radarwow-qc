import numpy as np

from radarwow.core.testing import synthetic_sweep
from radarwow.qc import Despeckle, Hubbert2009CMD, InterferenceSpikes, SignalMask, Vulpiani2012QualityMap


def _scores(flag, truth, valid):
    pod = (flag & truth).sum() / truth.sum()
    far_rate = (flag & ~truth & valid).sum() / (~truth & valid).sum()
    return pod, far_rate


def test_cmd_finds_clutter_patch():
    ds = synthetic_sweep(seed=1)
    out = Hubbert2009CMD()(ds, band="S")
    pod, fa = _scores(out["CMD_FLAG"].values, ds["CLUTTER_TRUTH"].values, np.isfinite(ds["DBZH"].values))
    assert pod > 0.8 and fa < 0.05, (pod, fa)


def test_quality_map_rejects_clutter():
    ds = synthetic_sweep(seed=2)
    out = Vulpiani2012QualityMap()(ds, band="C")
    pod, fa = _scores(out["QC_FLAG"].values, ds["CLUTTER_TRUTH"].values, np.isfinite(ds["DBZH"].values))
    assert pod > 0.8 and fa < 0.05, (pod, fa)


def test_interference_ray_detected():
    ds = synthetic_sweep(seed=3, clutter_patch=False, spike_az=123)
    out = InterferenceSpikes()(ds, band="C")
    rays = np.flatnonzero(out["INTERFERENCE_RAY"].values)
    assert list(rays) == [123]


def test_despeckle_and_signal():
    ds = synthetic_sweep(seed=4, clutter_patch=False)
    z = ds["DBZH"].values.copy()
    z[10, 500] = 30.0  # isolated speckle
    ds["DBZH"] = (("azimuth", "range"), z)
    out = Despeckle()(SignalMask()(ds, band="X"), band="X")
    assert np.isnan(out["DBZH"].values[10, 500])
    assert np.isfinite(out["DBZH"].values[90, 240])
