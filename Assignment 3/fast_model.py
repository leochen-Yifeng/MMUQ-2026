"""Numba-compiled copy of the supplied HBV001a model, plus objective functions.

The model equations are not rewritten: numba compiles the original
hmg.models.hbv001a_py._hbv001a function. check_equivalence() confirms that the
compiled version reproduces the pure-Python discharge.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from hmg import HBV001A
from hmg.models.hbv001a_py import _hbv001a

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'Rainfall-Runoff Modelling Data'

_model_nb = njit(cache=True)(_hbv001a)

# LnNSE: small offset so log() is defined even if simulated flow is zero.
LN_EPS = 1e-3


def load_data():
    data = pd.read_csv(DATA / 'time_series___24163005.csv', sep=';', index_col=0)
    data.index = pd.to_datetime(data.index, format='%Y-%m-%d-%H')
    area = float(pd.read_csv(DATA / 'area___24163005.csv', sep=';', index_col=0).iloc[0, 0])
    return data, area


class FastHBV:
    """Holds the fixed inputs; evaluate(prms) returns (NSE, LnNSE)."""

    def __init__(self):
        data, area = load_data()
        self.tems = data['tavg__ref'].to_numpy(np.float32)
        self.ppts = data['pptn__ref'].to_numpy(np.float32)
        self.pets = data['petn__ref'].to_numpy(np.float32)
        self.obs = data['diso__ref'].to_numpy(np.float64)
        self.dslr = np.float32(area / (1000.0 * 3600.0))
        n = len(self.tems)
        n_out = len(HBV001A().get_output_labels())
        self.otps = np.empty((n, n_out), np.float32)
        self.diss = np.empty(n, np.float32)
        self.den = np.sum((self.obs - self.obs.mean()) ** 2)
        lo = np.log(self.obs + LN_EPS)
        self.ln_obs = lo
        self.ln_den = np.sum((lo - lo.mean()) ** 2)

    def discharge(self, prms):
        _model_nb(self.tems, self.ppts, self.pets, self.otps, self.diss,
                  np.asarray(prms, np.float32), np.int64(1), self.dslr)
        return self.diss.astype(np.float64)

    def evaluate(self, prms):
        q = self.discharge(prms)
        if not np.isfinite(q).all():
            return np.nan, np.nan
        nse = 1.0 - np.sum((q - self.obs) ** 2) / self.den
        lq = np.log(np.maximum(q, 0.0) + LN_EPS)
        lnnse = 1.0 - np.sum((lq - self.ln_obs) ** 2) / self.ln_den
        return nse, lnnse


@njit(cache=True)
def _batch(tems, ppts, pets, otps, diss, P, dslr, obs, den, ln_obs, ln_den, eps, out):
    n = obs.shape[0]
    for r in range(P.shape[0]):
        _model_nb(tems, ppts, pets, otps, diss, P[r], np.int64(1), dslr)
        s1 = 0.0
        s2 = 0.0
        ok = True
        for t in range(n):
            q = np.float64(diss[t])
            if not np.isfinite(q):
                ok = False
                break
            s1 += (q - obs[t]) ** 2
            lq = np.log(max(q, 0.0) + eps)
            s2 += (lq - ln_obs[t]) ** 2
        if ok:
            out[r, 0] = 1.0 - s1 / den
            out[r, 1] = 1.0 - s2 / ln_den
        else:
            out[r, 0] = np.nan
            out[r, 1] = np.nan


def evaluate_batch(fm, P):
    """Rows of P are parameter vectors; returns array (rows, 2) = [NSE, LnNSE]."""
    P = np.ascontiguousarray(P, dtype=np.float32)
    out = np.empty((P.shape[0], 2))
    _batch(fm.tems, fm.ppts, fm.pets, fm.otps, fm.diss, P, fm.dslr, fm.obs,
           fm.den, fm.ln_obs, fm.ln_den, LN_EPS, out)
    return out


def check_equivalence(n=20, seed=0):
    """Compare numba vs. supplied pure-Python model on random parameter sets."""
    from calibration_config import PARAMETER_BOUNDS
    fm = FastHBV()
    ref = HBV001A()
    bounds = ref.get_parameter_bounds_in_correct_order(PARAMETER_BOUNDS)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        p = rng.uniform(bounds[:, 0], bounds[:, 1]).astype(np.float32)
        otps = np.empty_like(fm.otps); diss = np.empty_like(fm.diss)
        _hbv001a(fm.tems, fm.ppts, fm.pets, otps, diss, p, 1, fm.dslr)
        q = fm.discharge(p)
        worst = max(worst, float(np.max(np.abs(q - diss)) / max(1e-9, np.max(np.abs(diss)))))
    return worst


if __name__ == '__main__':
    import timeit
    print('max relative difference numba vs python:', check_equivalence())
    fm = FastHBV()
    p = np.array([0., .1, .01, .1, 0., 300., 70., 2.5, 0., 0., 1., .01, 30., 0., .1, .01, 1e-3, 1e-5])
    fm.evaluate(p)
    print('seconds per run:', timeit.timeit(lambda: fm.evaluate(p), number=200) / 200)
