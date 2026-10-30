"""Verify saved calibration artifacts and replay selected recorded candidates."""

import hashlib
import json

import numpy as np
import pandas as pd

from ab_calibrate_hbv import ROOT, create_model, load_data, simulate
from calibration_config import PARAMETER_BOUNDS, WARMUP_STEPS


def main():
    output = ROOT / 'calibration_outputs'
    summary = json.loads((output / 'run_summary.json').read_text())
    parameters = pd.read_csv(output / 'best_parameters.csv')
    history = pd.read_csv(output / 'evaluation_history.csv')
    generations = pd.read_csv(output / 'generation_history.csv')
    labels = parameters.parameter.tolist()
    data, area = load_data()
    model = create_model(data, area)
    bounds = model.get_parameter_bounds_in_correct_order(PARAMETER_BOUNDS)
    np.testing.assert_allclose(parameters[['lower_bound', 'upper_bound']], bounds)
    assert labels == sorted(model.get_parameter_labels(), key=model.get_parameter_labels().get)
    candidates = history[labels].to_numpy()
    assert np.isfinite(history.to_numpy()).all()
    assert (candidates >= bounds[:, 0]).all()
    assert (candidates <= bounds[:, 1]).all()
    assert len(history) == summary['function_evaluations']
    np.testing.assert_array_equal(history.evaluation, np.arange(1, len(history) + 1))
    np.testing.assert_allclose(history.best_so_far,
                               np.minimum.accumulate(history.objective_1_minus_nse))
    np.testing.assert_allclose(history.objective_1_minus_nse.min(), summary['best_objective'])
    assert len(generations) == summary['iterations']
    assert generations.function_evaluations.iloc[-1] == len(history)
    np.testing.assert_array_equal(generations.generation, np.arange(1, len(generations) + 1))
    for row in generations.itertuples():
        np.testing.assert_allclose(row.best_objective, history.best_so_far.iloc[row.function_evaluations - 1])

    for name, expected in summary['input_sha256'].items():
        actual = hashlib.sha256((ROOT / 'Rainfall-Runoff Modelling Data' / name).read_bytes()).hexdigest()
        assert actual == expected, f'Input data changed: {name}'

    observed = data['diso__ref'].to_numpy()[WARMUP_STEPS:]
    denominator = np.sum((observed - observed.mean()) ** 2)
    # Replay the best candidate plus samples across the history, not only metadata.
    indices = sorted(set([0, len(history) // 3, 2 * len(history) // 3,
                          len(history) - 1, int(history.objective_1_minus_nse.idxmin())]))
    for index in indices:
        q = simulate(model, candidates[index])
        objective = np.sum((q[WARMUP_STEPS:] - observed) ** 2) / denominator
        np.testing.assert_allclose(objective, history.objective_1_minus_nse.iloc[index], rtol=1e-6)

    best = simulate(model, parameters.best_value.to_numpy(), full_outputs=True)
    hydrographs = pd.read_csv(output / 'hydrographs.csv', index_col=0)
    np.testing.assert_allclose(best, hydrographs.calibrated_m3_s, rtol=1e-6)
    np.testing.assert_allclose(data['diso__ref'], hydrographs.observed_m3_s)
    np.testing.assert_array_equal(pd.to_datetime(hydrographs.index), data.index)
    objective = np.sum((best[WARMUP_STEPS:] - observed) ** 2) / denominator
    np.testing.assert_allclose(objective, summary['best_objective'], rtol=1e-6)
    outputs = model.get_outputs()
    np.testing.assert_allclose(outputs, pd.read_csv(output / 'internal_variables.csv', index_col=0),
                               rtol=1e-6, atol=1e-12)
    assert np.isfinite(outputs).all() and (best >= 0).all()

    population = np.load(output / 'final_population.npz')
    energies = population['objectives']
    np.testing.assert_allclose(energies.min(), summary['best_objective'])
    np.testing.assert_allclose(energies.std(), summary['population_objective_std'])
    threshold = summary['absolute_tolerance'] + summary['relative_tolerance'] * abs(energies.mean())
    if summary['status'] == 'converged':
        assert energies.std() <= threshold
    else:
        raise AssertionError('Calibration has not converged; do not label it complete.')

    for name in ['01_hydrographs.png', '02_objective_history.png',
                 '03_parameter_objective.png', '04_parameter_history.png',
                 '05_internal_storage.png', '06_model_diagnostics.png', 'assignment_report.md']:
        assert (output / name).stat().st_size > 0
    print(f'PASS: {len(history):,} evaluations, {len(generations)} generations, '
          f'{len(labels)} parameter bounds, convergence, input hashes, '
          'selected history replays, and complete final outputs.')


if __name__ == '__main__':
    main()
