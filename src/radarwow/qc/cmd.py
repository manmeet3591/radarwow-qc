"""Clutter Mitigation Decision (CMD), moment-level version of Hubbert, Dixon & Ellis (2009).

The paper's fuzzy logic uses TDBZ, SPIN, the clutter phase alignment (CPA) from I/Q time
series, and optional ZDR / PHIDP textures. Archived moments have no I/Q data, so CPA is not
available; the decision is made from the remaining features with the paper's weights,
renormalised. Detection is therefore weaker than the full CMD, especially for clutter
embedded in weather (the paper reports 50 % detection at CSR ~ -8 to -12 dB *with* CPA).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from radarwow.core import Algorithm, Reference, register
from radarwow.core.noise import signal_mask

from .textures import ramp, sd, sd_phase, spin, tdbz


@register
class Hubbert2009CMD(Algorithm):
    name = "hubbert2009_cmd"
    stage = "qc"
    reference = Reference("J. C. Hubbert, M. Dixon, S. M. Ellis", 2009,
                          "Weather Radar Ground Clutter. Part II: Real-Time Identification and Filtering",
                          "J. Atmos. Oceanic Technol. 26, 1181-1197", "10.1175/2009JTECHA1160.1")
    requires = ("DBZH",)
    optional = ("ZDR", "PHIDP", "SNRH", "SQIH")
    provides = ("CMD_INTEREST", "CMD_FLAG")
    notes = "No CPA (needs I/Q). Paper: TDBZ 20->40 dB^2, SPIN 15->30 %, SD(ZDR) -> 2.4 dB, SD(PHIDP) -> 15 deg; lower bounds of the ZDR/PHIDP ramps are our assumption."
    params = dict(tdbz_lo=20.0, tdbz_hi=40.0, spin_lo=15.0, spin_hi=30.0, spin_threshold_db=5.0,
                  sdzdr_lo=1.2, sdzdr_hi=2.4, sdphi_lo=8.0, sdphi_hi=15.0,
                  w_refl=1.0, w_zdr=0.5, w_phi=0.5, threshold=0.5, mask_output=False)

    def features(self, ds: xr.Dataset) -> dict:
        p = self.params
        z = ds["DBZH"].values.astype("float64")
        f = {"TDBZ": tdbz(z), "SPIN": spin(z, threshold_db=p["spin_threshold_db"])}
        if "ZDR" in ds:
            f["SD_ZDR"] = sd(ds["ZDR"].values.astype("float64"))
        if "PHIDP" in ds:
            f["SD_PHIDP"] = sd_phase(ds["PHIDP"].values.astype("float64"))
        return f

    def run_sweep(self, ds, cfg):
        p = self.params
        f = self.features(ds)
        refl = np.fmax(ramp(f["TDBZ"], p["tdbz_lo"], p["tdbz_hi"]), ramp(f["SPIN"], p["spin_lo"], p["spin_hi"]))
        num, den = p["w_refl"] * np.nan_to_num(refl), p["w_refl"] * np.isfinite(refl)
        if "SD_ZDR" in f:
            m = ramp(f["SD_ZDR"], p["sdzdr_lo"], p["sdzdr_hi"])
            num, den = num + p["w_zdr"] * np.nan_to_num(m), den + p["w_zdr"] * np.isfinite(m)
        if "SD_PHIDP" in f:
            m = ramp(f["SD_PHIDP"], p["sdphi_lo"], p["sdphi_hi"])
            num, den = num + p["w_phi"] * np.nan_to_num(m), den + p["w_phi"] * np.isfinite(m)
        with np.errstate(invalid="ignore", divide="ignore"):
            interest = np.where(den > 0, num / den, np.nan)
        sig = signal_mask(ds)
        flag = (interest > p["threshold"]) & sig
        out = ds.copy()
        out["CMD_INTEREST"] = (("azimuth", "range"), np.where(sig, interest, np.nan).astype("float32"),
                               {"long_name": "CMD clutter interest (0-1)", "units": "1"})
        out["CMD_FLAG"] = (("azimuth", "range"), flag, {"long_name": "clutter flag (Hubbert 2009 CMD, no CPA)"})
        for k, v in f.items():
            out[k] = (("azimuth", "range"), v.astype("float32"))
        if p["mask_output"]:
            for m in ("DBZH", "ZDR", "PHIDP", "RHOHV", "KDP"):
                if m in out:
                    out[m] = out[m].where(~flag)
        return out
