"""Signal mask and speckle removal (standard practice, no single source paper).

``SignalMask`` hides receiver-noise gates (SNR/NCP or noise floor, see radarwow.core.noise).
``Despeckle`` removes connected echo regions smaller than ``min_gates`` (8-connectivity,
azimuth wraps around 360 deg).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from radarwow.core import Algorithm, register
from radarwow.core.noise import signal_mask

MOMENTS = ("DBZH", "ZDR", "PHIDP", "RHOHV", "KDP", "VRADH", "WRADH")


@register
class SignalMask(Algorithm):
    name = "signal_mask"
    stage = "qc"
    requires = ("DBZH",)
    optional = ("SNRH", "SQIH")
    provides = ("SIGNAL",)
    params = dict(snr_min=3.0, sqi_min=0.4, margin_db=3.0, mask_output=True)

    def run_sweep(self, ds, cfg):
        p = self.params
        sig = signal_mask(ds, margin_db=p["margin_db"], snr_min=p["snr_min"], sqi_min=p["sqi_min"])
        out = ds.copy()
        out["SIGNAL"] = (("azimuth", "range"), sig)
        if p["mask_output"]:
            for m in MOMENTS:
                if m in out:
                    out[m] = out[m].where(sig)
        return out


@register
class Despeckle(Algorithm):
    name = "despeckle"
    stage = "qc"
    requires = ("DBZH",)
    provides = ("SPECKLE_FLAG",)
    params = dict(min_gates=10, mask_output=True)

    def run_sweep(self, ds, cfg):
        valid = np.isfinite(ds["DBZH"].values)
        n_az = valid.shape[0]
        tiled = np.concatenate([valid, valid[:1]], axis=0)  # wrap first ray to the end
        lab, n = ndimage.label(tiled, structure=np.ones((3, 3)))
        # merge labels across the 0/360 seam
        for a, b in zip(lab[0], lab[-1]):
            if a and b and a != b:
                lab[lab == b] = a
        lab = lab[:n_az]
        sizes = np.bincount(lab.ravel())
        small = (sizes < self.params["min_gates"])[lab] & valid
        out = ds.copy()
        out["SPECKLE_FLAG"] = (("azimuth", "range"), small)
        if self.params["mask_output"]:
            for m in MOMENTS:
                if m in out:
                    out[m] = out[m].where(~small)
        return out
