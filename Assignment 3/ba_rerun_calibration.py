"""Reproduce the ASGT-1 optimum with the same DE settings, using the fast model.

Only needed if calibration_outputs/best_parameters.csv from ASGT-1 is missing.
Writes calibration_outputs/best_parameters.csv in the same format as ASGT-1.
"""

import inspect

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from hmg import HBV001A
from calibration_config import (PARAMETER_BOUNDS, RANDOM_SEED, MAX_ITERATIONS,
                                POPULATION_MULTIPLIER, RELATIVE_TOLERANCE, POLISH)
from fast_model import FastHBV, ROOT


def main():
    fm = FastHBV()
    ref = HBV001A()
    labels = sorted(ref.get_parameter_labels(), key=ref.get_parameter_labels().get)
    bounds = ref.get_parameter_bounds_in_correct_order(PARAMETER_BOUNDS).astype(float)

    def objective(p):
        nse, _ = fm.evaluate(p)
        return 1.0 - nse

    rnd = 'rng' if 'rng' in inspect.signature(differential_evolution).parameters else 'seed'
    res = differential_evolution(
        objective, bounds, **{rnd: RANDOM_SEED}, maxiter=MAX_ITERATIONS,
        popsize=POPULATION_MULTIPLIER, tol=RELATIVE_TOLERANCE, polish=POLISH,
        workers=1, strategy='best1bin', updating='immediate', mutation=(0.5, 1.0),
        recombination=0.7, init='latinhypercube', atol=0.0)

    nse, lnnse = fm.evaluate(res.x)
    print(res.message, 'generations:', res.nit, 'evaluations:', res.nfev)
    print(f'NSE = {nse:.4f}, LnNSE = {lnnse:.4f}')
    out = ROOT / 'calibration_outputs'
    out.mkdir(exist_ok=True)
    pd.DataFrame({'parameter': labels, 'lower_bound': bounds[:, 0],
                  'upper_bound': bounds[:, 1], 'best_value': res.x,
                  'model_value_float32': res.x.astype(np.float32)}
                 ).to_csv(out / 'best_parameters.csv', index=False)


if __name__ == '__main__':
    main()
