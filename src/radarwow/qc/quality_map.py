"""Fuzzy quality index of Vulpiani et al. (2012), used to remove clutter, clear air and RLAN.

The paper combines trapezoidal memberships of a static clutter map, radial velocity and the
range textures of ZDR, rhoHV and PHIDP with weights 0.5 / 0.3 / 0.4 / 0.4 / 0.4 and rejects
gates with Q < 0.5. The trapezoid limits are not reproduced here from the paper; the ramps
below are our assumptions (documented in ``params``) and should be tuned per radar. The static
clutter map term is used only if a ``CLUTTER_MAP`` (0-1) variable is supplied.
"""

from __future__ import annotations

import numpy as np

from radarwow.core import Algorithm, Reference, register
from radarwow.core.noise import signal_mask

from .textures import ramp, sd, sd_phase


@register
class Vulpiani2012QualityMap(Algorithm):
    name = "vulpiani2012_quality"
    stage = "qc"
    reference = Reference("G. Vulpiani, M. Montopoli, L. Delli Passeri, A. G. Gioia, P. Giordano, F. S. Marzano", 2012,
                          "On the Use of Dual-Polarized C-Band Radar for Operational Rainfall Retrieval in Mountainous Areas",
                          "J. Appl. Meteor. Climatol. 51, 405-425", "10.1175/JAMC-D-10-05024.1")
    requires = ("DBZH", "RHOHV", "PHIDP")
    optional = ("ZDR", "VRADH", "CLUTTER_MAP")
    provides = ("QUALITY", "QC_FLAG")
    notes = "Weights from the paper; membership ramps are assumptions: SD(ZDR) 0.7->2 dB, SD(rhoHV) 0.05->0.15, SD(PHIDP) 6->15 deg, |V| 0.5->1.5 m/s."
    params = dict(w_map=0.5, w_vel=0.3, w_zdr=0.4, w_rho=0.4, w_phi=0.4, threshold=0.5,
                  sdzdr=(0.7, 2.0), sdrho=(0.05, 0.15), sdphi=(6.0, 15.0), vel=(0.5, 1.5), mask_output=False)

    def run_sweep(self, ds, cfg):
        p = self.params
        terms = []
        terms.append((p["w_rho"], 1 - ramp(sd(ds["RHOHV"].values.astype(float)), *p["sdrho"])))
        terms.append((p["w_phi"], 1 - ramp(sd_phase(ds["PHIDP"].values.astype(float)), *p["sdphi"])))
        if "ZDR" in ds:
            terms.append((p["w_zdr"], 1 - ramp(sd(ds["ZDR"].values.astype(float)), *p["sdzdr"])))
        if "VRADH" in ds:
            terms.append((p["w_vel"], ramp(np.abs(ds["VRADH"].values.astype(float)), *p["vel"])))
        if "CLUTTER_MAP" in ds:
            terms.append((p["w_map"], 1 - ds["CLUTTER_MAP"].values.astype(float)))
        num = sum(w * np.nan_to_num(t) for w, t in terms)
        den = sum(w * np.isfinite(t) for w, t in terms)
        with np.errstate(invalid="ignore", divide="ignore"):
            q = np.where(den > 0, num / den, np.nan)
        sig = signal_mask(ds)
        flag = sig & ~(q >= p["threshold"])
        out = ds.copy()
        out["QUALITY"] = (("azimuth", "range"), np.where(sig, q, np.nan).astype("float32"), {"long_name": "quality index Q (0-1)"})
        out["QC_FLAG"] = (("azimuth", "range"), flag, {"long_name": "rejected: Q < threshold (Vulpiani 2012)"})
        if p["mask_output"]:
            for m in ("DBZH", "ZDR", "PHIDP", "RHOHV", "KDP"):
                if m in out:
                    out[m] = out[m].where(sig & ~flag)
        return out
