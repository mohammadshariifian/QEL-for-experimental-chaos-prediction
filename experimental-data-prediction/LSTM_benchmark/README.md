# LSTM Benchmark for Time-Series Prediction

PyTorch LSTM benchmark for chaotic time-series prediction, orchestrated by Julia. Includes LSTM, Ridge Regression, and Naive Persistence baselines.

## Requirements

- **Julia** >= 1.10 with packages: `CSV`, `DataFrames`, `JSON`
- **Python** >= 3.8 with packages: `torch`, `numpy`
- Install Julia deps: `julia --project -e 'using Pkg; Pkg.add(["CSV", "DataFrames", "JSON"])'`
- Install Python deps: `pip install torch numpy`

## Data Format

Place your data file at `data/data.csv` (relative to the project root where you run the script).

Format: CSV with no header, each row is one variable (time series), columns are time steps.

Example for 4 variables with 1000 time steps:
```
val1_t1,val1_t2,...,val1_t1000
val2_t1,val2_t2,...,val2_t1000
val3_t1,val3_t2,...,val3_t1000
val4_t1,val4_t2,...,val4_t1000
```

## Usage

### Run all variables (1-to-1 mode)
```bash
bash LSTM_benchmark/run_lstm_benchmark.sh
```

### Run a specific variable
```bash
bash LSTM_benchmark/run_lstm_benchmark.sh 1   # variable 1 only
```

### Run different modes
```bash
julia LSTM_benchmark/lstm_benchmark.jl 1       # 1-to-1 (single input, single output)
julia LSTM_benchmark/lstm_benchmark.jl 3       # 3-to-1 (3 inputs, 1 output)
julia LSTM_benchmark/lstm_benchmark.jl 4       # 4-to-1 (4 inputs, 1 output)
```

### Run specific variable in a specific mode
```bash
julia LSTM_benchmark/lstm_benchmark.jl 1 2     # mode=1, variable 2 only
```

## Hyperparameter Grid

| Parameter | Values |
|-----------|--------|
| Lookback  | 1, 3, 5, 7, 9 |
| Hidden Units | 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64 |
| Layers | 1, 2 |
| Seeds | 42, 43, 44, 45, 46 |

Total configs per variable: 5 x 13 x 2 x 5 = 650

## Output

Results are saved to `results_lstm/`:
- `1to1/var1/` - per-variable JSON results and `summary_metrics.csv`
- `1to1_summary_all.csv` - combined summary across all variables

Each JSON result contains: RMSE, MAE, NRMSE, R-squared for LSTM, Ridge, and Naive Persistence.

## Customization

To adapt for different data:

1. **Number of variables**: Edit `N_VARS` in `lstm_benchmark.jl` and the `n_vars` parameter in Python scripts
2. **Variable names**: Edit `VAR_IDS` in `lstm_benchmark.jl`
3. **Downsampling interval**: Edit `past_intervals` in `lstm_single.py` / `lstm_single_multi.py` (default: 1, no downsampling)
4. **Train/test split ratio**: Edit the `0.7` ratio in Python scripts (default: 70% train, 30% test)
5. **Hyperparameter grid**: Edit `LOOKBACKS`, `HIDDEN_UNITS`, `N_LAYERS`, `SEEDS` in `lstm_benchmark.jl`
6. **Multi-variable input configs**: Edit `INPUT_IDS_3TO1` and `INPUT_IDS_4TO1` in `lstm_benchmark.jl`

## File Structure

```
LSTM_benchmark/
├── README.md                  # This file
├── run_lstm_benchmark.sh      # Entry point shell script
├── lstm_benchmark.jl          # Julia orchestrator (multi-threaded)
├── lstm_single.py             # Single-variable PyTorch LSTM (1-to-1)
└── lstm_single_multi.py       # Multi-variable PyTorch LSTM (1-to-1, 3-to-1, 4-to-1)
```
