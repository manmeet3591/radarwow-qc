"""Interference (RLAN / emitter) spike detection.

Saltikoff et al. (2016) describe the signature: an external emitter adds constant received power,
so the echo fills the whole ray (rising as 20 log r in dBZ), is noise-like (low rhoHV, random
PHIDP/ZDR), and is intermittent from scan to scan. Their paper is a review; this detector turns
that signature into a ray test. Rays are flagged, gates on them that still look like weather
(high rhoHV) are kept.
"""

from __future__ import annotations

import warnings

import numpy as np

from radarwow.core import Algorithm, Reference, register
from radarwow.core.noise import signal_mask


@register
class InterferenceSpikes(Algorithm):
    name = "interference_spikes"
    stage = "qc"
    reference = Reference("E. Saltikoff, J. Y. N. Cho, P. Tristant, A. Huuskonen, L. Allmon, R. Cook, E. Becker, P. Joe", 2016,
                          "The Threat to Weather Radars by Wireless Technology", "Bull. Amer. Meteor. Soc. 97, 1159-1167",
                          "10.1175/BAMS-D-15-00048.1")
    requires = ("DBZH",)
    optional = ("RHOHV", "SNRH", "SQIH")
    provides = ("INTERFERENCE_RAY", "INTERFERENCE_FLAG")
    notes = "Ray test derived from the signature described in the review (signal-filled ray, noise-like rhoHV)."
    params = dict(r_start_km=30.0, min_fill=0.9, fill_excess=0.4, rhohv_max=0.85, mask_output=False)

    def run_sweep(self, ds, cfg):
        p = self.params
        sig = signal_mask(ds)
        r = ds["range"].values / 1e3
        far = r > p["r_start_km"]
        fill = sig[:, far].mean(axis=1) if far.any() else np.zeros(sig.shape[0])
        ray = (fill > p["min_fill"]) & (fill > np.median(fill) + p["fill_excess"])
        gate = np.zeros_like(sig)
        if "RHOHV" in ds:
            rho = ds["RHOHV"].values
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rmed = np.nanmedian(np.where(sig[:, far], rho[:, far], np.nan), axis=1) if far.any() else np.ones(sig.shape[0])
            ray &= rmed < p["rhohv_max"]
            gate = ray[:, None] & sig & ~(rho > p["rhohv_max"])
        else:
            gate = ray[:, None] & sig
        out = ds.copy()
        out["INTERFERENCE_RAY"] = (("azimuth",), ray)
        out["INTERFERENCE_FLAG"] = (("azimuth", "range"), gate)
        if p["mask_output"]:
            for m in ("DBZH", "ZDR", "PHIDP", "RHOHV", "KDP"):
                if m in out:
                    out[m] = out[m].where(~gate)
        return out
