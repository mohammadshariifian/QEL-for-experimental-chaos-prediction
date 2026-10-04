# Quantum Reservoir Computing for Chaotic Circuit Time-Series Prediction

Code and data accompanying the paper (4-variable chaotic circuit prediction with a
quantum reservoir computer, plus LSTM baselines and peak/extrema prediction).

This repository contains **minimal, directly runnable code** and the **results data**
needed to reproduce every figure and table in the paper.

---

## 1. Prerequisites

- **Julia 1.12.2** (pinned in `Manifest.toml`)
- **Python >= 3.8** with `torch` and `numpy` (LSTM baseline only)
- **LaTeX** with the `newtx` package (PGFPlotsX high-quality figure output)

### Sibling packages

`src/head.jl` loads two packages relative to this repository:

```julia
push!(LOAD_PATH, "../QuantumCircuits/src", "../VQC/src")
```

Clone them next to this repository:

```bash
git clone https://github.com/guochu/QuantumCircuits.git   ../QuantumCircuits
git clone https://github.com/guochu/VQC.jl.git            ../VQC
```

> The experiment scripts also add `../QuantumReservoirComputing/src` to `LOAD_PATH`.
> That package is **not** actually imported (all reservoir helpers live in `src/`),
> so it may be omitted.

### Install dependencies

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
pip install torch numpy
```

---

## 2. Data provided

| Path | Description |
|------|-------------|
| `data/data_theory.csv` | Raw chaotic circuit simulation, 4 rows (`uC1,uC2,iL1,iL2`) x 93750 columns |
| `data/peaks_all{,_max,_min}.csv` | Pre-extracted extrema (all / maxima / minima) used by the peak experiments |
| `data/generate_timeseries.jl`, `data/generate_timeseries.m` | Circuit ODE generators (Julia / MATLAB) |
| `result_4to1_v2/linear_taun/{var}_{past}/` | 4-to-1 RMSE grid (`results.jld2`) + the single best prediction per target (`pred_*.jld2`) |
| `result_4to1_v2/linear_taun/{uC1_34,iL1_68}/` | Matched-interval companion predictions used by the cross-variable attractor figures (see below) |
| `result_1to1_v2/linear_taun/{var}_{past}/results.jld2` | 1-to-1 RMSE grids (appendix) |
| `result_4to1_v3/` | Best-config mean/std over 100 random Hamiltonians (`stds.jld2`, `stds_summary.csv`) |
| `results_peak/local{max,min}/{var}/results.jld2` | Peak-prediction RMSE grids |
| `results_lstm_v2/` | LSTM baseline summaries (aggregated CSV + reports) |

`PAST_INTERVALS = uC1 -> 21, uC2 -> 7, iL1 -> 34, iL2 -> 68`.

**Cross-attractor companion data.** To draw a cross-variable attractor `x(t)` vs `y(t)`,
both variables must be predicted with the *same* past interval and the *same* best
`(N_steps, N_hid)` configuration, so the two trajectories are directly comparable:

| Cross pair | Directories | Interval (x, y) | Best config |
|------------|-------------|-----------------|-------------|
| uC1 - iL1 | `iL1_34` + `uC1_34` | (34, 34) | (1, 7), (1, 7) |
| iL2 - uC2 | `iL2_68` + `uC2_7` | (68, 7) | (2, 3), (1, 7) |
| iL2 - iL1 | `iL2_68` + `iL1_68` | (68, 68) | (2, 3), (2, 3) |

`uC1_34` and `iL1_68` each contain their `results.jld2` plus the single matched
prediction file; the canonical `uC1_21` / `iL1_34` / `iL2_68` / `uC2_7` directories
are used for the single-variable figures (Fig 2-4). `plot_attractor_pgfplotsx.jl`
encodes this pairing in its `CROSS_PAST` constant.

---

## 3. Reproduce the figures (from the included results)

Run from this directory:

```bash
# Figure 2 - 4-to-1 RMSE heatmaps
julia --project=. plot_heatmap_pgfplotsx.jl

# Figure 3 - prediction time-series
( cd result_4to1_v2 && julia --project=.. plot_results.jl )

# Figure 4 - attractor reconstruction
julia --project=. plot_attractor_pgfplotsx.jl

# Figure 5 - peak-detection RMSE heatmaps
julia --project=. plot_peak_heatmap_pgfplotsx.jl
```

Or simply: `bash scripts/reproduce_figures.sh`

Output is written to `paper_figures/` (see `paper_figures/README.md` for the full
figure inventory).

---

## 4. Regenerate the results from the raw data

These are computationally expensive; the pre-computed results above are provided so
that the figures can be regenerated without re-running the experiments.

```bash
# Fig 2-4 data: 4-to-1 reservoir, all target variables
julia --project=. Time_serial_circuit_4to1_v2.jl linear tau_nqubit 11 0 1:3 1:10

# Fig 5 data: peak (extrema) prediction, all variables, both peak types
julia --project=. Time_serial_circuit_peak.jl

# Paper table: best-config RMSE mean +/- std over 100 random Hamiltonians
julia --project=. Time_serial_circuit_4to1_v3.jl

# Appendix: 1-to-1 reservoir, all target variables
julia --project=. Time_serial_circuit_1to1_v2.jl linear tau_nqubit 11 0 1:10 1:10

# LSTM baseline (writes results_lstm_v2/)
bash LSTM/run_lstm_benchmark.sh
```

---

## 5. Figure -> script -> data map

| Figure / table | Generation script | Source data | Produced by |
|----------------|-------------------|-------------|-------------|
| Fig 2 RMSE heatmaps | `plot_heatmap_pgfplotsx.jl` | `result_4to1_v2/linear_taun/*/results.jld2` | `Time_serial_circuit_4to1_v2.jl` |
| Fig 3 prediction curves | `result_4to1_v2/plot_results.jl` | above + best `pred_*.jld2` + `data/data_theory.csv` | `Time_serial_circuit_4to1_v2.jl` |
| Fig 4 attractors | `plot_attractor_pgfplotsx.jl` | best `pred_*.jld2` + `data/data_theory.csv` | `Time_serial_circuit_4to1_v2.jl` |
| Fig 5 peak heatmaps | `plot_peak_heatmap_pgfplotsx.jl` | `results_peak/local{max,min}/*/results.jld2` | `Time_serial_circuit_peak.jl` |
| RMSE mean +/- std table | — | `result_4to1_v3/stds.jld2` | `Time_serial_circuit_4to1_v3.jl` |
| Appendix 1-to-1 grids | `plot_heatmap_pgfplotsx.jl` | `result_1to1_v2/linear_taun/*/results.jld2` | `Time_serial_circuit_1to1_v2.jl` |
| LSTM baseline | — | `results_lstm_v2/*.csv` | `LSTM/lstm_benchmark.jl` + `LSTM/lstm_single{,_multi}.py` |

---

## 6. Repository layout

```
.
├── src/                                  # core reservoir / Hamiltonian helpers
│   ├── head.jl                           # loads sibling packages + includes the rest
│   ├── auxiliary.jl                      # operator overloads
│   ├── core.jl                           # Ham / Ham_fc / Ham_XYZ_nn builders
│   └── circuitQR.jl                      # encoding, reservoir evolution, training
├── data/                                 # raw data + generators
├── Time_serial_circuit_4to1_v2.jl        # main 4-to-1 experiment
├── Time_serial_circuit_1to1_v2.jl        # 1-to-1 experiment (appendix)
├── Time_serial_circuit_peak.jl           # peak/extrema experiment
├── Time_serial_circuit_4to1_v3.jl        # mean/std over 100 Hamiltonians
├── results_peak/results_peak_plot.jl     # helpers included by the peak experiment
├── plot_heatmap_pgfplotsx.jl             # Fig 2 (and appendix)
├── plot_attractor_pgfplotsx.jl           # Fig 4
├── plot_peak_heatmap_pgfplotsx.jl        # Fig 5
├── result_4to1_v2/plot_results.jl        # Fig 3
├── LSTM/                                 # LSTM / ridge / naive baselines
├── paper_figures/                        # generated figures + description
└── scripts/                              # helper scripts
```

The per-configuration LSTM result JSONs (5k+ small files) are **not** stored here;
the aggregated `summary_metrics.csv` / `*_summary_all.csv` contain the same metrics
and can be regenerated with `LSTM/lstm_benchmark.jl`.

---

## Notes

- `uC1_34` and `iL1_68` are **not** legacy data: they are the matched-interval
  companions required for the cross-variable attractor figures (see §2).
- `data/peaks_all*.csv` were produced externally from `data/data_theory.csv`; the
  extrema-extraction step is not part of this repository.
