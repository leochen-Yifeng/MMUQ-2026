"""Assignment 3: variance-based global sensitivity analysis (Sobol indices).

Cases: {full range, narrow range (optimum +-20 %)} x {NSE, LnNSE}.
Y is the objective function value of one model run (a scalar), not the hydrograph.

Sampling (Smith 2014, Algorithm 15.10; Saltelli et al. 2010):
    A, B  : N x p quasi-random (scrambled Sobol) matrices
    AB_i  : A with column i taken from B           (Saltelli's A_B^(i))
    -> N * (p + 2) model runs per range. NSE and LnNSE come from the same runs.

Estimators (V = var(Y) from y_A and y_B together):
    Saltelli 2010, Table 2 (b): S_i  = mean(y_B * (y_ABi - y_A)) / V
    Saltelli 2010, Table 2 (f): S_Ti = 0.5 * mean((y_A - y_ABi)^2) / V   (Jansen)
    Smith 2014, Eqs. 15.24-15.26 (written for C_i = B with column i from A; by
    symmetry the same formulas hold with A and B swapped, so AB_i is reused):
        f0^2 = mean(y_A) * mean(y_B)
        S_i  = (mean(y_B * y_ABi) - f0^2) / (mean(y_B^2) - f0^2)
        S_Ti = 1 - (mean(y_A * y_ABi) - f0^2) / (mean(y_A^2) - f0^2)
"""

from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import qmc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hmg import HBV001A
from calibration_config import PARAMETER_BOUNDS
from fast_model import FastHBV, evaluate_batch, check_equivalence, ROOT

OUT = ROOT / 'sobol_outputs'
NARROW = 0.20  # +- 20 % around the ASGT-1 optimum
OFS = ['NSE', 'LnNSE']


# ----------------------------------------------------------------------------
# Ranges
# ----------------------------------------------------------------------------

def get_ranges(labels):
    ref = HBV001A()
    full = ref.get_parameter_bounds_in_correct_order(PARAMETER_BOUNDS).astype(float)
    best = pd.read_csv(ROOT / 'calibration_outputs' / 'best_parameters.csv')
    assert list(best['parameter']) == labels, 'Parameter order mismatch.'
    x = best['best_value'].to_numpy(float)
    lo = np.minimum(x * (1 - NARROW), x * (1 + NARROW))  # negative x flips the order
    hi = np.maximum(x * (1 - NARROW), x * (1 + NARROW))
    narrow = np.column_stack([np.clip(lo, full[:, 0], full[:, 1]),
                              np.clip(hi, full[:, 0], full[:, 1])])
    return full, narrow, x


# ----------------------------------------------------------------------------
# Sampling and model runs
# ----------------------------------------------------------------------------

def sample(bounds, n, seed):
    p = len(bounds)
    u = qmc.Sobol(d=2 * p, scramble=True, seed=seed).random(n)
    A = qmc.scale(u[:, :p], bounds[:, 0], bounds[:, 1])
    B = qmc.scale(u[:, p:], bounds[:, 0], bounds[:, 1])
    return A, B


def run_all(fm, A, B):
    """Returns yA (N,2), yB (N,2), yAB (p,N,2)."""
    p = A.shape[1]
    yA = evaluate_batch(fm, A)
    yB = evaluate_batch(fm, B)
    yAB = np.empty((p,) + yA.shape)
    for i in range(p):
        ABi = A.copy()
        ABi[:, i] = B[:, i]
        yAB[i] = evaluate_batch(fm, ABi)
    return yA, yB, yAB


# ----------------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------------

def _center(yA, yB, yAB):
    # Subtracting the mean does not change the indices in theory, but it removes
    # large cancellation errors when mean(Y) >> std(Y) (e.g. NSE ~ 0.89 +- 0.01
    # in the narrow range). Without it the first-order CIs explode.
    m = np.mean(np.concatenate([yA, yB]))
    return yA - m, yB - m, yAB - m


def saltelli(yA, yB, yAB):
    yA, yB, yAB = _center(yA, yB, yAB)
    V = np.var(np.concatenate([yA, yB]))
    S = np.mean(yB * (yAB - yA), axis=1) / V
    ST = 0.5 * np.mean((yA - yAB) ** 2, axis=1) / V
    return S, ST


def smith(yA, yB, yAB):
    yA, yB, yAB = _center(yA, yB, yAB)
    f02 = np.mean(yA) * np.mean(yB)
    S = (np.mean(yB * yAB, axis=1) - f02) / (np.mean(yB ** 2) - f02)
    ST = 1 - (np.mean(yA * yAB, axis=1) - f02) / (np.mean(yA ** 2) - f02)
    return S, ST


def bootstrap_ci(yA, yB, yAB, n_boot=300, seed=1):
    rng = np.random.default_rng(seed)
    n = len(yA)
    S_b, ST_b = [], []
    for _ in range(n_boot):
        j = rng.integers(0, n, n)
        s, st = saltelli(yA[j], yB[j], yAB[:, j])
        S_b.append(s); ST_b.append(st)
    S_b, ST_b = np.array(S_b), np.array(ST_b)
    return (np.percentile(S_b, [2.5, 97.5], axis=0),
            np.percentile(ST_b, [2.5, 97.5], axis=0))


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

def bar_plot(ax, labels, S, ST, ciS=None, ciST=None, title=''):
    x = np.arange(len(labels))
    w = 0.38
    kw = dict(capsize=2, ecolor='0.3', error_kw=dict(lw=0.7))
    eS = None if ciS is None else np.abs(ciS - S)
    eST = None if ciST is None else np.abs(ciST - ST)
    ax.bar(x - w / 2, S, w, yerr=eS, label='First order $S_i$', color='tab:blue', **kw)
    ax.bar(x + w / 2, ST, w, yerr=eST, label='Total effect $S_{T_i}$', color='tab:orange', **kw)
    ax.set_xticks(x, labels, rotation=90, family='monospace', fontsize=8)
    ax.set_ylabel('Sensitivity [-]')
    ax.set_title(title, fontsize=10)
    ax.axhline(0, color='k', lw=0.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log2n', type=int, default=16, help='N = 2**log2n base samples')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--reuse', action='store_true', help='reuse saved model runs')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    print('numba vs. python model, max rel. diff:', check_equivalence(5))
    fm = FastHBV()
    ref = HBV001A()
    labels = sorted(ref.get_parameter_labels(), key=ref.get_parameter_labels().get)
    full, narrow, x_opt = get_ranges(labels)
    nse0, lnnse0 = fm.evaluate(x_opt)
    print(f'Optimum: NSE={nse0:.4f}, LnNSE={lnnse0:.4f}')

    pd.DataFrame({'parameter': labels, 'optimum': x_opt,
                  'full_lo': full[:, 0], 'full_hi': full[:, 1],
                  'narrow_lo': narrow[:, 0], 'narrow_hi': narrow[:, 1]}
                 ).to_csv(OUT / 'ranges.csv', index=False)

    N = 2 ** args.log2n
    p = len(labels)
    results = {}   # (range, of) -> dict
    conv_rows = []
    summary = {'N': N, 'p': p, 'model_runs_per_range': N * (p + 2),
               'optimum_NSE': nse0, 'optimum_LnNSE': lnnse0}

    for rname, bounds in [('full', full), ('narrow', narrow)]:
        t0 = time.perf_counter()
        cache = OUT / f'runs_{rname}.npz'
        if args.reuse and cache.exists():
            d = np.load(cache)
            A, B, yA, yB, yAB = d['A'], d['B'], d['yA'], d['yB'], d['yAB']
            assert len(A) == N, 'Cached runs have a different N.'
            print(f'{rname}: reusing cached model runs')
        else:
            A, B = sample(bounds, N, args.seed)
            yA, yB, yAB = run_all(fm, A, B)
            np.savez_compressed(cache, A=A, B=B, yA=yA, yB=yB, yAB=yAB)
        dt = time.perf_counter() - t0
        print(f'{rname}: {N * (p + 2):,} runs in {dt:.1f} s')

        for k, of in enumerate(OFS):
            a, b, ab = yA[:, k], yB[:, k], yAB[:, :, k]
            good = np.isfinite(a) & np.isfinite(b) & np.isfinite(ab).all(axis=0)
            a, b, ab = a[good], b[good], ab[:, good]
            S, ST = saltelli(a, b, ab)
            S_sm, ST_sm = smith(a, b, ab)
            ciS, ciST = bootstrap_ci(a, b, ab)
            yall = np.concatenate([a, b])
            results[(rname, of)] = dict(S=S, ST=ST, ciS=ciS, ciST=ciST, y=yall)
            summary[f'{rname}_{of}'] = {
                'valid_rows': int(good.sum()), 'mean_Y': float(yall.mean()),
                'var_Y': float(yall.var()), 'min_Y': float(yall.min()),
                'max_Y': float(yall.max()), 'median_Y': float(np.median(yall)),
                'sum_S': float(S.sum()), 'sum_ST': float(ST.sum()),
                'sec': dt}
            pd.DataFrame({
                'parameter': labels,
                'S_saltelli': S, 'S_ci_lo': ciS[0], 'S_ci_hi': ciS[1],
                'ST_saltelli': ST, 'ST_ci_lo': ciST[0], 'ST_ci_hi': ciST[1],
                'S_smith': S_sm, 'ST_smith': ST_sm,
                'rank_ST': (-ST).argsort().argsort() + 1,
            }).to_csv(OUT / f'indices_{rname}_{of}.csv', index=False)

            # Convergence: Sobol prefixes of length 2^m are balanced point sets.
            for m in range(8, args.log2n + 1):
                n = 2 ** m
                s, st = saltelli(a[:n], b[:n], ab[:, :n])
                for j, lbl in enumerate(labels):
                    conv_rows.append(dict(range=rname, of=of, N=n, parameter=lbl,
                                          S=s[j], ST=st[j]))

        # Scatter plots of Y vs. each parameter (cf. Smith 2014, Fig. 15.1).
        for k, of in enumerate(OFS):
            fig, axs = plt.subplots(3, 6, figsize=(18, 8.5), sharey=True)
            n_sc = min(4000, N)
            y = yA[:n_sc, k]
            for j, ax in enumerate(axs.flat):
                ax.plot(A[:n_sc, j], y, '.', ms=1.5, alpha=0.35)
                ax.axvline(x_opt[j], color='r', lw=0.8)
                ax.set_xlabel(labels[j], family='monospace')
                ax.grid(alpha=0.3)
            if rname == 'full':
                axs[0, 0].set_ylim(max(np.nanpercentile(y, 1), -5), 1)
            for ax in axs[:, 0]:
                ax.set_ylabel(of)
            fig.suptitle(f'{of} vs. parameters, {rname} range ({n_sc} samples of A; red = optimum)')
            save(fig, f'scatter_{rname}_{of}.png')

    conv = pd.DataFrame(conv_rows)
    conv.to_csv(OUT / 'convergence.csv', index=False)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))

    # 1) One bar chart per case.
    for (rname, of), r in results.items():
        fig, ax = plt.subplots(figsize=(10, 4.5))
        bar_plot(ax, labels, r['S'], r['ST'], r['ciS'], r['ciST'],
                 f'{of}, {rname} range, N = {N:,} (whiskers: 95 % bootstrap CI)')
        save(fig, f'bars_{rname}_{of}.png')

    # 2) 2 x 2 overview.
    fig, axs = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
    for i, rname in enumerate(['full', 'narrow']):
        for k, of in enumerate(OFS):
            r = results[(rname, of)]
            bar_plot(axs[i, k], labels, r['S'], r['ST'], r['ciS'], r['ciST'],
                     f'{of}, {rname} range')
    fig.suptitle(f'Sobol indices (Saltelli et al. 2010), N = {N:,}, '
                 f'{N * (p + 2):,} model runs per range')
    save(fig, 'bars_overview.png')

    # 3) Total-effect comparison across all four cases.
    fig, ax = plt.subplots(figsize=(12, 4.8))
    xx = np.arange(p)
    w = 0.2
    for c, ((rname, of), r) in enumerate(results.items()):
        ax.bar(xx + (c - 1.5) * w, r['ST'], w, label=f'{of}, {rname}')
    ax.set_xticks(xx, labels, rotation=90, family='monospace', fontsize=8)
    ax.set_ylabel('Total effect $S_{T_i}$ [-]')
    ax.set_title('Total-effect indices: effect of range and objective function')
    ax.grid(alpha=0.3); ax.legend()
    save(fig, 'total_effect_comparison.png')

    # 4) Convergence of ST for the top parameters in each case.
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for i, rname in enumerate(['full', 'narrow']):
        for k, of in enumerate(OFS):
            ax = axs[i, k]
            top = np.argsort(-results[(rname, of)]['ST'])[:5]
            sub = conv[(conv['range'] == rname) & (conv['of'] == of)]
            for j in top:
                d = sub[sub['parameter'] == labels[j]]
                l, = ax.plot(d['N'], d['ST'], '-o', ms=3, label=f'{labels[j]} $S_T$')
                ax.plot(d['N'], d['S'], '--', color=l.get_color(), lw=0.9)
            ax.set_xscale('log', base=2)
            ax.set_title(f'{of}, {rname} range (solid $S_T$, dashed $S_i$)', fontsize=10)
            ax.grid(alpha=0.3); ax.legend(fontsize=7)
            ax.set_xlabel('N'); ax.set_ylabel('Index')
    save(fig, 'convergence.png')

    # 5) Distribution of Y in each case.
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    for i, rname in enumerate(['full', 'narrow']):
        for k, of in enumerate(OFS):
            y = results[(rname, of)]['y']
            ax = axs[i, k]
            lo = np.percentile(y, 0.5)
            ax.hist(np.clip(y, lo, None), bins=100)
            ax.set_title(f'{of}, {rname} range (clipped at 0.5th pct. = {lo:.2f})', fontsize=9)
            ax.set_xlabel(of); ax.grid(alpha=0.3)
    save(fig, 'output_distributions.png')

    # Text summary
    for (rname, of), r in results.items():
        order = np.argsort(-r['ST'])
        top = ', '.join(f'{labels[j]} ({r["S"][j]:.2f}/{r["ST"][j]:.2f})' for j in order[:6])
        print(f'{of:6s} {rname:6s} sumS={r["S"].sum():.2f} sumST={r["ST"].sum():.2f} | top S/ST: {top}')


if __name__ == '__main__':
    main()
