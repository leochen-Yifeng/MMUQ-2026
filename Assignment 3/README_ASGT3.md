# ASGT-3: Sobol global sensitivity analysis of HBV001a

## Files

| File | Purpose |
|---|---|
| `fast_model.py` | Compiles the supplied `hmg` model function with numba, unchanged (about 100× faster; max. relative difference from pure Python ≈ 1e-6, float32 rounding). Computes NSE and LnNSE together. |
| `ba_rerun_calibration.py` | Only needed if ASGT-1's `calibration_outputs/best_parameters.csv` is missing. Re-runs DE with exactly the ASGT-1 settings and writes that file. |
| `bb_sobol_gsa.py` | ASGT-3 itself: 4 cases = {full, narrow ±20 %} × {NSE, LnNSE}. |

Put these files in the ASGT-1 folder, next to `hmg/`, `calibration_config.py` and `Rainfall-Runoff Modelling Data/`.

```bash
pip install numba
python ba_rerun_calibration.py          # skip this if you already have best_parameters.csv
python bb_sobol_gsa.py --log2n 16       # N = 65,536, about 4-5 min
python bb_sobol_gsa.py --log2n 16 --reuse   # re-plot only, without re-running the model
```

## Method

- Y = objective function value of one model run (NSE or LnNSE). Using 1−NSE gives identical indices.
- A and B are scrambled Sobol quasi-random samples. AB_i = A with column i taken from B.
  This needs N(p+2) = 65,536 × 20 = 1.31 M model runs per range. NSE and LnNSE are computed from the same runs.
- Saltelli et al. 2010, Table 2: S_i from (b), S_Ti from (f) (Jansen). Smith 2014, Eqs. 15.24–15.26 are computed as a cross-check (`S_smith`, `ST_smith` in the CSVs); the two agree.
- Y is centred before estimation. In theory this changes nothing, but in the narrow range NSE ≈ 0.89 ± 0.01. Without centring, the first-order 95 % CIs were as wide as ±0.8.
- 95 % confidence intervals come from bootstrapping (300 resamples). Convergence is checked for N = 2^8 … 2^16.
- Narrow range = x_opt·(1 ± 0.2), clipped to the full bounds (this clips `urr_wsr` and `lrr_dre` at the upper end).

## Results (N = 65,536; first-order S / total effect ST)

The narrow range is built around the ASGT-1 optimum in `Assignment 1/calibration_outputs/best_parameters.csv` (NSE = 0.907).

| Case | Most sensitive (S / ST) | ΣS | ΣST |
|---|---|---|---|
| NSE, full | urr_tdh (0.07/0.65), sl0_fcy (0.02/0.58), snw_att (0.01/0.52), urr_ulc (0.01/0.26), urr_tdr (0.00/0.22), sl0_bt0 (0.01/0.17) | 0.13 | 2.66 |
| LnNSE, full | sl0_fcy (0.29/0.44), sl0_bt0 (0.29/0.43), lrr_lct (0.11/0.16), snw_att (0.07/0.10), lrr_dre (0.03/0.06) | 0.81 | 1.21 |
| NSE, narrow | sl0_fcy (0.42/0.53), lrr_dre (0.29/0.38), snw_pmf (0.10/0.15), sl0_dth (0.01/0.05), snw_amf (0.00/0.04) | 0.83 | 1.17 |
| LnNSE, narrow | lrr_dre (0.81/0.82), sl0_fcy (0.12/0.15), sl0_bt0 (0.01/0.03), sl0_dth (0.01/0.02) | 0.97 | 1.03 |

## Points for the discussion

1. **NSE, full range: interactions dominate.** ΣS ≈ 0.13 but ΣST ≈ 2.7. Almost all the variance of NSE comes from parameter
   combinations, for example where the snow threshold, the soil capacity and the URR threshold together
   decide whether a flood peak is produced at all. NSE is also heavy-tailed (median −0.65, minimum −34).
   A few very bad runs dominate var(Y), so this case converges slowest. Its CIs are about ±0.1 even at N = 65,536; see `convergence.png`.
2. **LnNSE, full range: much more additive** (ΣS ≈ 0.8). The log puts weight on low flows, so the parameters that
   control runoff volume and baseflow take over: soil `sl0_fcy`, `sl0_bt0`, and the LRR loss `lrr_lct`. The peak-flow
   parameters `urr_tdh` and `urr_tdr` become irrelevant.
3. **Narrow range: the ranking changes.** Near the optimum, NSE is controlled by `sl0_fcy`, `lrr_dre` and `snw_pmf`, and LnNSE almost
   entirely by `lrr_dre` (baseflow recession). `urr_tdh` and `snw_att` drop out. This is partly a property of the ±20 % design itself: the narrow range of each parameter
   is proportional to its optimum value. `snw_att` ≈ −0.006 °C gets a range only about 0.0025 °C wide, and
   `urr_ndr`, `urr_uct`, `lrr_lct` (≈ 1e-5) get almost no range. A zero index here means "not sensitive
   *within this range*", not "unimportant".
4. **Parameters that are irrelevant everywhere** (ST ≈ 0 in all four cases): `snw_dth`, `sl0_pwp`, `urr_dth`, `lrr_dth`,
   `urr_wsr`, `urr_ndr`, `urr_uct`. These could be fixed. This can be compared with the ASGT-2 local SA.
5. **Comparison with ASGT-2 (local OAT, ±20 %).** The local top 3 were `sl0_fcy`, `snw_pmf`, `lrr_dre`. These are exactly the
   global top 3 for NSE in the narrow range, so the local and global results agree near the optimum. ST ≈ S there (little
   interaction), so the one-at-a-time analysis is adequate in this region.
   In the full range the picture is different: `urr_tdh` and `urr_tdr` had **zero** local sensitivity (near the optimum the
   URR store never exceeds the threshold `urr_tdh` ≈ 98 mm, so the upper outlet is inactive). Globally, however, they are among
   the most influential NSE parameters, through interactions. This is the main argument for global SA.
6. **Equifinality.** Re-running DE gave NSE = 0.907 with clearly different values for `sl0_pwp` (52 vs. 646) and `urr_tdh` (165 vs. 98).
   Both parameters have ST ≈ 0 near the optimum, which is consistent with this.
7. **Limitations:** the Sobol decomposition assumes independent, uniformly distributed parameters. In reality the calibrated
   parameters are correlated (e.g. `sl0_fcy`–`sl0_bt0`).

## Outputs (`sobol_outputs/`)

`bars_overview.png` (2×2), `bars_<range>_<OF>.png`, `total_effect_comparison.png`, `convergence.png`,
`scatter_<range>_<OF>.png` (cf. Smith Fig. 15.1), `output_distributions.png`,
`indices_<range>_<OF>.csv`, `ranges.csv`, `summary.json`.
