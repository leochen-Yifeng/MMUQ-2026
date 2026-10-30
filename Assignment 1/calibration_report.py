"""Write a readable assignment report from the actual calibration results."""

import numpy as np


def write_report(output, summary, parameters, data, best):
    """Keep the report consistent with the saved parameter and result tables."""
    example = summary['example']
    calibrated = summary['calibrated']
    position = (parameters.best_value - parameters.lower_bound) / (
        parameters.upper_bound - parameters.lower_bound)
    near_bound = parameters.loc[(position <= 0.01) | (position >= 0.99), 'parameter'].tolist()
    rows = '\n'.join(
        f'| `{row.parameter}` | {row.lower_bound:.6g} | {row.upper_bound:.6g} | {row.best_value:.9g} |'
        for row in parameters.itertuples()
    )
    observed = data['diso__ref'].to_numpy()
    peak_index = int(np.argmax(observed))
    peak_text = (
        f'At the largest observed peak ({data.index[peak_index]}), observed discharge '
        f'is {observed[peak_index]:.2f} m3/s and simulated discharge is '
        f'{best[peak_index]:.2f} m3/s. These are compared at the same timestamp; '
        'the simulated maximum may occur at another time.'
    )
    status_text = (
        'SciPy reports successful convergence under the population-spread criterion.'
        if summary['status'] == 'converged' else
        'The iteration limit was reached before convergence; this result is provisional.'
    )
    report = f'''# MMUQ Assignment 01: Calibration of HBV001a

## 1. Aim and supplied material

The aim is to find a parameter vector that reproduces the supplied observed hydrograph
using HBV001a and SciPy Differential Evolution (DE). The model implementation is unchanged.
The 18 parameter bounds were transcribed by parameter name from the supplied Moodle
screenshot, `sketch (1).png`. This implementation uses the accompanying hourly record
and catchment area file. Input file SHA-256 hashes are stored in `run_summary.json`.

The record contains {len(data):,} hourly steps. The evaluation period is
{summary['evaluation_start']} to {summary['evaluation_end']}.
Catchment area is {summary['catchment_area_m2'] / 1e6:.1f} km2.
Temperature, precipitation and potential evapotranspiration drive the model.
Observed discharge is used to evaluate its output, not as a forcing input.
Runoff depth in mm per hour is converted to m3/s using area / (1000 * 3600).

## 2. Objective function and repeatable model evaluation

For simulated discharge Qsim(theta) and observed discharge Qobs, define

J(theta) = 1 - NSE(theta)
         = sum((Qsim(theta) - Qobs)^2) / sum((Qobs - mean(Qobs))^2).

DE minimizes J. A perfect fit has J = 0 and NSE = 1. NSE = 0 corresponds to the
squared error of a constant prediction equal to the observed mean; NSE can be negative.
The observed denominator is calculated once on the evaluation period.

All {len(parameters)} supplied parameter entries are included in the vector, including
four initial storages. Every candidate restarts the model from its own specified initial
storages. During optimization only the discharge series is needed. The best candidate
is rerun with full internal outputs for diagnostics. Mode consistency, restart behavior,
the objective formula, and copied history records are checked by `--check`.
The model casts parameters to float32; `best_parameters.csv` includes both the optimizer
values and the effective float32 values.

No warm-up is removed ({summary['warmup_steps']} steps), and the entire record is used for
calibration. There is no held-out validation period. These are explicit implementation
choices because the supplied written instructions do not specify a temporal split.
The reported performance therefore describes fitting this record, not predictive skill
on independent data. Squared errors emphasize larger discharge errors.

## 3. Optimization and stopping condition

The method is SciPy `differential_evolution`, strategy `best1bin`, with seed
{summary['seed']}, population multiplier {summary['population_multiplier']}, Latin-hypercube
initialization, mutation interval (0.5, 1.0), recombination 0.7, immediate updating,
one worker, relative tolerance {summary['relative_tolerance']}, absolute tolerance 0,
and a maximum of {summary['max_iterations']} generations. Local polishing is disabled so
that the recorded search describes DE alone. These settings are implementation choices.

{status_text} The run used {summary['iterations']} generations and
{summary['function_evaluations']:,} objective evaluations in {summary['elapsed_seconds']:.1f} seconds.
The final population objective standard deviation is
{summary['population_objective_std']:.8g}; its convergence threshold is
{summary['convergence_threshold']:.8g}.
The check is std(population objective) <= atol + tol * abs(mean(population objective)).
Convergence measures agreement within this population; it does not prove global
optimality, parameter uniqueness, or physical identifiability. This is one seeded run.

Every objective call appends a copy of its parameter vector and objective value.
`evaluation_history.csv` contains the complete history, rather than only accepted
population members or per-generation best candidates. `generation_history.csv` contains
one summary per generation. The final population is saved in `final_population.npz`.

## 4. Results

| Metric | Original demonstration | Calibrated |
|---|---:|---:|
| NSE | {example['nse']:.6f} | {calibrated['nse']:.6f} |
| RMSE [m3/s] | {example['rmse_m3_s']:.6f} | {calibrated['rmse_m3_s']:.6f} |
| Mean bias [m3/s] | {example['mean_bias_m3_s']:.6f} | {calibrated['mean_bias_m3_s']:.6f} |
| Volume bias [%] | {example['volume_bias_percent']:.4f} | {calibrated['volume_bias_percent']:.4f} |

Positive bias means overestimation. The best objective value is
{summary['best_objective']:.9g}. The original demonstration is a before-calibration
reference only: some of its values lie outside the supplied calibration bounds, so it
was not used as DE's initial candidate and is not a feasible baseline for this search.

![Observed, original and calibrated hydrographs](01_hydrographs.png)

{peak_text}

### Best parameter vector

| Parameter | Lower bound | Upper bound | Best optimizer value |
|---|---:|---:|---:|
{rows}

Parameters within 1% of a supplied bound: {', '.join(near_bound) or 'none'}.
A boundary result is reported without expanding the supplied search region.
It can reflect limited data, parameter trade-offs, or the chosen bounds and objective;
it does not alone establish that the physical parameter is known precisely.
Names and ordering follow the actual model interface; generic HBV parameter tables
and ambiguous screenshot comments are not used to redefine these parameters.

## 5. Search history and interpretation

![Every objective evaluation and running best](02_objective_history.png)

The upper panel shows all attempted objective values. Individual trials can get worse
while the search improves. The lower panel shows the running minimum, which can only
decrease or remain constant. After the first candidate it improved
{summary['number_of_best_updates']} times; flat portions are intervals without a new best.
When every objective value is positive, the upper panel and parameter/objective scatter
plots use a logarithmic objective axis to show both poor trials and the better region.
No evaluations are discarded from the plots or the history table.

![Each parameter versus objective](03_parameter_objective.png)

Each panel plots one parameter against the objective for every evaluated candidate.
The red star marks the final best vector. A low-objective region indicates values seen
in good sampled combinations, not an isolated causal effect of that parameter.
The plots are exploratory diagnostics rather than a formal sensitivity analysis or
a posterior distribution.

![Parameter trajectories](04_parameter_history.png)

Blue points show all attempted parameter values in evaluation order; red lines show
the corresponding coordinate of the best vector found so far. Parameter coordinates
can move in either direction even though the best objective never increases.
Broad spread or a changing best coordinate can indicate weak constraint or trade-offs.

## 6. Internal diagnostics and limitations

![Internal storage](05_internal_storage.png)

![Forcing and complete internal diagnostics](06_model_diagnostics.png)

All saved internal outputs finite: {summary['all_outputs_finite']}.
Minimum simulated discharge: {summary['minimum_discharge_m3_s']:.8g} m3/s.
Maximum absolute hourly water-balance residual:
{summary['max_absolute_balance_residual_mm']:.8g} mm.
Small balance residuals check numerical conservation, not goodness of fit.
The supplied implementation's `get_discharge()` uses the channel outflow. Separate
subsurface exports are shown in the diagnostics and included by the supplied balance
calculation; this calibration retains that original discharge definition.

This solution covers the requirements in the supplied one-page instructions:
parameter-vector evaluation, DE with supplied bounds, an explicit objective, complete
PV/OFV tracking, parameter-versus-objective plots, parameter and objective trajectories,
the running best, and presentation of the final result. Submission format, deadline,
grading criteria and any additional classroom requirements were not supplied. They
are not invented here. This package is ready for review against those course details.

## 7. Reproduction and reading order

Use Python 3.12 and the package versions listed in `requirements.txt` in the parent
directory. From that directory run `python ab_calibrate_hbv.py --check`, then
`python ab_calibrate_hbv.py`. On macOS, `run_calibration.command` uses `python3`, or `PYTHON_EXECUTABLE` if set,
after dependencies have been installed. A full run overwrites the final output
directory. The earlier 100-generation backup is stored outside this upload folder.

Read `calibration_config.py` first, then `main()` in `ab_calibrate_hbv.py`, followed by
`load_data`, `create_model`, `simulate`, `make_objective` and `save_results`.
`calibration_report.py` formats this report. The supplied model and original
demonstration are preserved. No new forcing data or model parameters are fabricated.
'''
    (output / 'assignment_report.md').write_text(report, encoding='utf-8')
