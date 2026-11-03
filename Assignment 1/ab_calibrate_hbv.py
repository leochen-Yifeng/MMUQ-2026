"""Calibrate HBV001a, save every evaluation, and plot the search history.

Read in this order: main -> load_data -> simulate -> objective -> save_results.
The supplied hmg implementation and the original demonstration are unchanged.
"""

from contextlib import redirect_stdout
from pathlib import Path
import argparse
import io
import json
import hashlib
import inspect
import time

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import differential_evolution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hmg import HBV001A
from calibration_config import (
    PARAMETER_BOUNDS, WARMUP_STEPS, RANDOM_SEED, MAX_ITERATIONS,
    POPULATION_MULTIPLIER, RELATIVE_TOLERANCE, POLISH,
)
from calibration_report import write_report

# 以脚本所在目录为基准寻找数据和输出，不依赖启动时所在的目录。
ROOT = Path(__file__).resolve().parent


def load_data():
    """Read the supplied hourly forcing, observations, and catchment area."""
    # 时间序列包含气象输入和观测流量；面积保存在另一份文件中。
    folder = ROOT / 'Rainfall-Runoff Modelling Data'
    data = pd.read_csv(folder / 'time_series___24163005.csv', sep=';', index_col=0)
    data.index = pd.to_datetime(data.index, format='%Y-%m-%d-%H')
    # 检查数值、时间顺序和逐小时步长，避免用错误的数据计算目标函数。
    columns = ['tavg__ref', 'pptn__ref', 'petn__ref', 'diso__ref']
    if not np.isfinite(data[columns].to_numpy()).all():
        raise ValueError('Input or observed data contain non-finite values.')
    if not data.index.is_unique or not data.index.is_monotonic_increasing:
        raise ValueError('Timestamps must be unique and increasing.')
    if not (data.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError('The discharge conversion requires hourly time steps.')
    area = float(pd.read_csv(folder / 'area___24163005.csv', sep=';', index_col=0).iloc[0, 0])
    if not np.isfinite(area) or area <= 0:
        raise ValueError('Catchment area must be positive and finite.')
    return data, area


def create_model(data, area):
    """Prepare fixed inputs once; candidate parameters change during DE."""
    # 模型只创建一次：气象输入固定，优化时反复替换的是参数向量。
    model = HBV001A()
    # 观测流量用于评价模型，不作为驱动模型的气象输入。
    model.set_inputs(
        data['tavg__ref'].to_numpy(dtype=float),
        data['pptn__ref'].to_numpy(dtype=float),
        data['petn__ref'].to_numpy(dtype=float),
    )
    model.set_outputs(len(data))
    # Convert runoff depth in mm per hour to discharge in cubic metres per second.
    # 面积单位为 m²：mm/h × 面积 / (1000 × 3600) 得到 m³/s。
    model.set_discharge_scaler(area / (1000.0 * 3600.0))
    return model


def simulate(model, parameters, full_outputs=False):
    """Run from the initial storage values in this candidate parameter vector."""
    # 每组参数包含初始储量；模型每次从该组初始条件重新运行。
    model.set_parameters(np.asarray(parameters, dtype=float))
    # 优化时只保留必要输出；最终诊断时保存完整内部变量。
    model.set_optimization_flag(0 if full_outputs else 1)
    # The supplied wrapper prints a runtime for every call; suppress that noise.
    with redirect_stdout(io.StringIO()):
        model.run_model()
    discharge = model.get_discharge().copy()
    if not np.isfinite(discharge).all():
        raise FloatingPointError('The candidate produced non-finite discharge.')
    return discharge


def make_objective(model, observed, warmup, history):
    """Build J(theta) = 1 - NSE and record every function evaluation."""
    if not 0 <= warmup < len(observed):
        raise ValueError('Warm-up must leave at least one evaluation step.')
    # 去除预热期，并提前计算 NSE 的固定分母；本项目 warmup 为 0。
    reference = observed[warmup:]
    denominator = np.sum((reference - reference.mean()) ** 2)
    if denominator <= 0:
        raise ValueError('NSE requires non-constant observed discharge.')

    # 闭包记住模型、观测值和历史列表，优化器只需要传入参数向量。
    def objective(parameters):
        # J = 1 - NSE = 模拟误差平方和 / 观测值离差平方和，越小越好。
        simulated = simulate(model, parameters)
        residual = simulated[warmup:] - reference
        value = float(np.sum(residual ** 2) / denominator)
        # Copy the vector: an optimizer may later reuse or modify its arrays.
        history.append((np.asarray(parameters).copy(), value))
        return value

    return objective


def save_results(data, area, model, labels, bounds, history, generations,
                 result, baseline, best, elapsed, output):
    """Export reproducible tables, diagnostics, and assignment figures."""
    # 将每次尝试过的参数、目标值和截至当时的最优值整理成表格。
    output.mkdir(parents=True, exist_ok=True)
    vectors = np.array([item[0] for item in history])
    values = np.array([item[1] for item in history])
    # 累积最小值只会下降或保持不变，表示每次评估后找到的最好结果。
    running_best = np.minimum.accumulate(values)
    table = pd.DataFrame(vectors, columns=labels)
    table.insert(0, 'evaluation', np.arange(1, len(table) + 1))
    table['objective_1_minus_nse'] = values
    table['best_so_far'] = running_best
    table.to_csv(output / 'evaluation_history.csv', index=False)
    pd.DataFrame(generations).to_csv(output / 'generation_history.csv', index=False)
    # Saving the final population also preserves the optimizer's final state.
    np.savez_compressed(output / 'final_population.npz',
                        parameters=result.population, objectives=result.population_energies)

    # 同时保存优化器给出的参数和模型内部实际使用的 float32 值。
    parameters = pd.DataFrame({
        'parameter': labels, 'lower_bound': bounds[:, 0],
        'upper_bound': bounds[:, 1], 'best_value': result.x,
        # HBV001a internally casts candidate parameters to float32.
        'model_value_float32': result.x.astype(np.float32),
    })
    parameters.to_csv(output / 'best_parameters.csv', index=False)
    pd.DataFrame({
        'observed_m3_s': data['diso__ref'],
        'example_m3_s': baseline, 'calibrated_m3_s': best,
    }, index=data.index).to_csv(output / 'hydrographs.csv')
    # 此时模型刚用最优参数完整运行过，读取对应的内部状态。
    outputs = model.get_outputs().copy()
    output_labels = model.get_output_labels()
    ordered_outputs = sorted(output_labels, key=output_labels.get)
    pd.DataFrame(outputs, index=data.index, columns=ordered_outputs).to_csv(
        output / 'internal_variables.csv')

    observed = data['diso__ref'].to_numpy()[WARMUP_STEPS:]
    denominator = np.sum((observed - observed.mean()) ** 2)
    # 使用同一个评价时段计算 NSE、RMSE、平均偏差和水量偏差。
    def metrics(flow):
        residual = flow[WARMUP_STEPS:] - observed
        return {'nse': float(1 - np.sum(residual ** 2) / denominator),
                'rmse_m3_s': float(np.sqrt(np.mean(residual ** 2))),
                'mean_bias_m3_s': float(np.mean(residual)),
                'volume_bias_percent': float(100 * np.sum(residual) / np.sum(observed))}
    # 汇总设置、结果和输入文件哈希，便于检查和复现此次运行。
    summary = {
        'status': 'converged' if result.success else 'stopped_without_convergence',
        'message': str(result.message), 'iterations': int(result.nit),
        'function_evaluations': int(result.nfev), 'seed': RANDOM_SEED,
        'max_iterations': MAX_ITERATIONS, 'population_multiplier': POPULATION_MULTIPLIER,
        'relative_tolerance': RELATIVE_TOLERANCE, 'polish': POLISH,
        'absolute_tolerance': 0.0, 'strategy': 'best1bin',
        'updating': 'immediate', 'workers': 1,
        'mutation': [0.5, 1.0], 'recombination': 0.7, 'init': 'latinhypercube',
        'population_objective_std': float(np.std(result.population_energies)),
        'convergence_threshold': float(RELATIVE_TOLERANCE * abs(np.mean(result.population_energies))),
        'scipy_version': scipy.__version__, 'numpy_version': np.__version__,
        'warmup_steps': WARMUP_STEPS, 'validation': 'No held-out validation period',
        'evaluation_start': str(data.index[WARMUP_STEPS]),
        'evaluation_end': str(data.index[-1]), 'catchment_area_m2': area,
        'example': metrics(baseline), 'calibrated': metrics(best),
        'best_objective': float(result.fun), 'elapsed_seconds': elapsed,
        'max_absolute_balance_residual_mm': float(np.max(np.abs(outputs[:, output_labels['mod_bal']]))),
        'minimum_discharge_m3_s': float(best.min()),
        'all_outputs_finite': bool(np.isfinite(outputs).all()),
        'number_of_best_updates': int(np.count_nonzero(np.diff(running_best) < 0)),
        'bounds_source': 'Supplied sketch (1).png; all 18 entries transcribed by parameter name',
        'input_sha256': {
            name: hashlib.sha256((ROOT / 'Rainfall-Runoff Modelling Data' / name).read_bytes()).hexdigest()
            for name in ['time_series___24163005.csv', 'area___24163005.csv']
        },
    }
    (output / 'run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    # 图 1：比较观测、原始示例和校准后的流量过程。
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(data.index, data['diso__ref'], label='Observed', color='black', lw=1)
    ax.plot(data.index, baseline, label='Original example', alpha=0.5, lw=0.8)
    ax.plot(data.index, best, label='Calibrated', alpha=0.85, lw=0.9)
    ax.set(xlabel='Time', ylabel='Discharge [m3/s]', title='Calibration on the available record')
    ax.legend(); ax.grid(alpha=0.25)
    save_figure(fig, output / '01_hydrographs.png')

    # 图 2：上方为每次尝试，下方为截至该次评估的最优目标值。
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    steps = np.arange(1, len(values) + 1)
    axes[0].plot(steps, values, '.', markersize=1.5, alpha=0.45)
    axes[0].set(ylabel='1 - NSE', title='Every objective evaluation')
    if np.all(values > 0):
        axes[0].set_yscale('log')
        axes[0].set_ylabel('1 - NSE (log scale)')
    axes[1].plot(steps, running_best)
    axes[1].set(xlabel='Function evaluation', ylabel='Best 1 - NSE so far')
    for ax in axes: ax.grid(alpha=0.25)
    save_figure(fig, output / '02_objective_history.png')

    # 图 3 和图 4：分别查看参数与目标值的关系、参数随评估次数的变化。
    for kind in ['scatter', 'trajectory']:
        fig, axes = plt.subplots(6, 3, figsize=(14, 18))
        for j, ax in enumerate(axes.flat):
            if kind == 'scatter':
                ax.scatter(vectors[:, j], values, s=2, alpha=0.25)
                ax.scatter(result.x[j], result.fun, marker='*', color='red', s=70)
                ax.set(xlabel=labels[j], ylabel='1 - NSE')
                if np.all(values > 0):
                    ax.set_yscale('log')
            else:
                ax.plot(steps, vectors[:, j], '.', markersize=1, alpha=0.3)
                # Also show the parameter belonging to the best solution so far.
                # 找到每个时刻历史最优解所在的评估序号，再取对应参数。
                best_indices = np.maximum.accumulate(np.where(
                    np.r_[True, np.diff(running_best) < 0], np.arange(len(values)), 0))
                ax.plot(steps, vectors[best_indices, j], color='red', lw=0.8)
                ax.set(xlabel='Function evaluation', ylabel=labels[j])
            ax.grid(alpha=0.2)
        fig.suptitle('Parameters versus objective' if kind == 'scatter' else
                     'Parameter search history (red: best solution so far)')
        save_figure(fig, output / ('03_parameter_objective.png' if kind == 'scatter' else
                                  '04_parameter_history.png'))

    # 图 5：检查雪、土壤、上水库和下水库的储量随时间的变化。
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for name, ax in zip(['snw_dth', 'sl0_dth', 'urr_dth', 'lrr_dth'], axes):
        ax.plot(data.index, outputs[:, output_labels[name]])
        ax.set_ylabel(name + ' [mm]'); ax.grid(alpha=0.25)
    axes[-1].set_xlabel('Time')
    fig.suptitle('Internal storage with calibrated parameters')
    save_figure(fig, output / '05_internal_storage.png')

    # Complete diagnostics follow the variables shown in the supplied example.
    fig, axes = plt.subplots(8, 1, figsize=(12, 16), sharex=True)
    axes[0].plot(data.index, data['tavg__ref']); axes[0].set_ylabel('Temperature [C]')
    axes[1].plot(data.index, data['pptn__ref']); axes[1].set_ylabel('Precipitation [mm/step]')
    axes[2].plot(data.index, outputs[:, output_labels['snw_dth']]); axes[2].set_ylabel('Snow [mm]')
    axes[3].plot(data.index, outputs[:, output_labels['sl0_dth']]); axes[3].set_ylabel('Soil water [mm]')
    axes[4].plot(data.index, data['petn__ref'], label='Potential')
    axes[4].plot(data.index, outputs[:, output_labels['sl0_etn']], label='Actual')
    axes[4].set_ylabel('ET [mm/step]'); axes[4].legend()
    for name in ['urr_dth', 'lrr_dth']:
        axes[5].plot(data.index, outputs[:, output_labels[name]], label=name)
    axes[5].set_ylabel('Storage [mm]'); axes[5].legend()
    axes[6].plot(data.index, outputs[:, output_labels['chn_pow']], label='Channel')
    axes[6].plot(data.index, outputs[:, output_labels['urr_urf']] +
                 outputs[:, output_labels['lrr_lrf']], label='Subsurface export')
    axes[6].set_ylabel('Outflow [mm/step]'); axes[6].legend()
    axes[7].plot(data.index, outputs[:, output_labels['mod_bal']])
    axes[7].set(xlabel='Time', ylabel='Balance residual [mm]')
    for ax in axes: ax.grid(alpha=0.25)
    fig.suptitle('Inputs and internal variables of the calibrated HBV001a model')
    save_figure(fig, output / '06_model_diagnostics.png')

    # Generate the interpretation from the same numeric results as the figures.
    write_report(output, summary, parameters, data, best)
    print(json.dumps(summary, indent=2))


def save_figure(fig, path):
    # Reserve space above grids so the shared title cannot overlap a panel.
    fig.tight_layout(rect=(0, 0, 1, 0.97) if fig._suptitle else (0, 0, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    # --check 只检查模型和目标函数，不执行耗时的完整优化。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true',
                        help='Check objective, restart behavior, and output modes without optimizing.')
    args = parser.parse_args()
    # 先读取固定输入，再创建模型并按模型要求的顺序整理参数边界。
    data, area = load_data()
    model = create_model(data, area)
    # 参数名、边界和向量位置必须一一对应，否则会给参数套错范围。
    labels = sorted(model.get_parameter_labels(), key=model.get_parameter_labels().get)
    bounds = model.get_parameter_bounds_in_correct_order(PARAMETER_BOUNDS)
    if not np.isfinite(bounds).all():
        raise ValueError('DE requires finite search bounds.')
    # history 保存每次评估；generations 保存每一代的摘要。
    history = []
    generations = []
    # 构造优化器需要的接口：传入参数向量，返回一个误差值。
    objective = make_objective(model, data['diso__ref'].to_numpy(), WARMUP_STEPS, history)

    # Original demonstration values, used only as a before-calibration reference.
    # Some values lie outside the supplied search bounds; do not use them as x0.
    example = np.array([0., .1, .01, .1, 0., 300., 70., 2.5, 0., 0.,
                        1., .01, 30., 0., .1, .01, 1e-3, 1e-5])
    baseline = simulate(model, example, full_outputs=True)

    # 用相同参数重复运行，检查初始状态、输出模式和目标函数。
    if args.check:
        midpoint = bounds.mean(axis=1)
        first = simulate(model, midpoint)
        simulate(model, bounds[:, 0])
        repeated = simulate(model, midpoint, full_outputs=True)
        np.testing.assert_allclose(first, repeated, rtol=0, atol=0)
        value = objective(midpoint)
        expected = np.sum((repeated[WARMUP_STEPS:] - data['diso__ref'].to_numpy()[WARMUP_STEPS:]) ** 2)
        observed = data['diso__ref'].to_numpy()[WARMUP_STEPS:]
        expected /= np.sum((observed - observed.mean()) ** 2)
        np.testing.assert_allclose(value, expected)
        assert len(history) == 1 and not np.shares_memory(history[0][0], midpoint)
        print('Checks passed: objective formula, fresh initial conditions, output modes, history copy.')
        return

    # 回调函数由 SciPy 在每代结束时调用，记录种群的收敛情况。
    started = time.perf_counter()
    def track_generation(intermediate_result):
        """Record generation statistics; every PV is recorded in objective()."""
        energies = intermediate_result.population_energies
        generations.append({
            'generation': intermediate_result.nit,
            'function_evaluations': len(history),
            'best_objective': float(intermediate_result.fun),
            'population_objective_std': float(np.std(energies)),
            'convergence_threshold': float(RELATIVE_TOLERANCE * abs(np.mean(energies))),
        })
        if intermediate_result.nit == 1 or intermediate_result.nit % 25 == 0:
            print(f'Generation {intermediate_result.nit}: '
                  f'best NSE = {1 - intermediate_result.fun:.6f}', flush=True)

    print(f'Starting DE: {len(labels)} parameters, {len(data)} hourly steps.', flush=True)
    # DE 在参数边界内搜索；每个候选都会调用 objective 并记录历史。
    # 随机种子和其他设置来自配置文件；polish=False 表示不追加局部优化。
    # SciPy 1.15 起使用 rng；旧版使用 seed。根据当前环境选择接口。
    random_argument = 'rng' if 'rng' in inspect.signature(differential_evolution).parameters else 'seed'
    result = differential_evolution(
        objective, bounds, **{random_argument: RANDOM_SEED}, maxiter=MAX_ITERATIONS,
        popsize=POPULATION_MULTIPLIER, tol=RELATIVE_TOLERANCE,
        polish=POLISH, workers=1, callback=track_generation,
        strategy='best1bin', updating='immediate', mutation=(0.5, 1.0),
        recombination=0.7, init='latinhypercube', atol=0.0,
    )
    # result.x 是找到的最优参数：重新运行以保存全部内部变量。
    best = simulate(model, result.x, full_outputs=True)
    # Re-evaluation must agree with the best objective recorded by SciPy.
    observed = data['diso__ref'].to_numpy()[WARMUP_STEPS:]
    recomputed = np.sum((best[WARMUP_STEPS:] - observed) ** 2) / np.sum((observed - observed.mean()) ** 2)
    np.testing.assert_allclose(recomputed, result.fun, rtol=1e-6)
    assert len(history) == result.nfev
    # 输出参数、历史、流量、诊断图和报告。
    save_results(data, area, model, labels, bounds, history, generations, result, baseline, best,
                 time.perf_counter() - started, ROOT / 'calibration_outputs')


# 直接运行此文件时启动 main；被其他脚本导入时不自动执行。
if __name__ == '__main__':
    main()
