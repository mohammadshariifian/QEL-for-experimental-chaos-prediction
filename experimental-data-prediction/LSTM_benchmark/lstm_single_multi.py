import torch
import numpy as np
import csv
import os
import sys
import json

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def load_data(path):
    with open(path, "r", newline="") as f:
        rows = [list(map(float, row)) for row in csv.reader(f)]

    data = np.asarray(rows, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError(f"Expected a 2D CSV array, received shape {data.shape}")

    return data

def zscore_normalize(data, train_start, train_end):
    train_data = data[:, train_start:train_end]
    mu = np.mean(train_data, axis=1, keepdims=True)
    sigma = np.std(train_data, axis=1, keepdims=True)
    sigma[sigma < 1e-10] = 1.0
    return (data - mu) / sigma

def make_windows_multi(
    data_vecs,
    target_vec,
    lookback,
    past_interval,
    range_start,
    range_end,
    allow_history_before_start=False,
):
    n_vars = len(data_vecs)

    # Training:
    # first target occurs only after enough history exists
    # inside the training interval.
    if not allow_history_before_start:
        first_target = range_start + lookback * past_interval

    # Testing:
    # allow history to come from immediately before test_start,
    # so every lookback begins predicting at the same test point.
    else:
        first_target = range_start

    target_indices = np.arange(
        first_target,
        range_end + 1,
        1,
        dtype=int,
    )

    # Safety: ensure enough preceding samples exist.
    valid = (
        target_indices - lookback * past_interval >= 0
    )
    target_indices = target_indices[valid]

    n_obs = len(target_indices)

    if n_obs <= 0:
        return None, None

    X = np.zeros(
        (n_obs, lookback, n_vars),
        dtype=np.float64,
    )
    Y = np.zeros(
        n_obs,
        dtype=np.float64,
    )

    for i, target_idx in enumerate(target_indices):

        history_indices = (
            target_idx
            - np.arange(
                lookback,
                0,
                -1,
                dtype=int,
            ) * past_interval
        )

        for v, vec in enumerate(data_vecs):
            X[i, :, v] = vec[history_indices]

        Y[i] = target_vec[target_idx]

    return X, Y


def compute_metrics(y_true, y_pred):
    err = y_pred - y_true
    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    s = np.std(y_true)
    nrmse_std = rmse / s if s > 1e-10 else rmse
    sst = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1.0 - np.sum(err**2) / sst if sst > 1e-10 else float('nan')
    return rmse, mae, nrmse_std, r2

def main():
    cfg = json.loads(sys.argv[1])
    data_file = cfg['data_file']
    mode = cfg['mode']
    output_id = cfg['output_id']
    lookback = cfg['lookback']
    hidden = cfg['hidden']
    n_layers = cfg['n_layers']
    seed = cfg['seed']
    output_dir = cfg['output_dir']

    n_vars = cfg.get("n_vars", 4)

    var_name_list = cfg.get(
        "var_names",
        [f"var{i+1}" for i in range(n_vars)],
    )

    if len(var_name_list) != n_vars:
        raise ValueError(
            f"Expected {n_vars} variable names, "
            f"received {len(var_name_list)}"
        )

    var_names = {
        i + 1: var_name_list[i]
        for i in range(n_vars)
    }

    past_interval_list = cfg["past_intervals"]

    if len(past_interval_list) != n_vars:
        raise ValueError(
            f"Expected {n_vars} past intervals, "
            f"received {len(past_interval_list)}"
        )

    past_intervals = {
        i + 1: int(past_interval_list[i])
        for i in range(n_vars)
    }
    output_name = var_names[output_id]
    past_interval = past_intervals[output_id]



    if mode == 3:
        input_ids = [1, 3, 4]
    elif mode == 4:
        input_ids = [1, 2, 3, 4]
    else:
        raise ValueError(
            f"Unknown mode {mode}. Only 3-to-1 and 4-to-1 are supported."
        )

    dropout = 0.1
    lr = 1e-3
    epochs = 200
    batch_size = 256
    ridge_lambda = 1e-6

    raw = load_data(data_file)
    n_total = raw.shape[1]

    if all(
        key in cfg
        for key in (
            "train_start_idx",
            "train_end_idx",
            "test_start_idx",
            "test_end_idx",
        )
    ):
        train_start = int(cfg["train_start_idx"])
        train_end = int(cfg["train_end_idx"])
        test_start = int(cfg["test_start_idx"])
        test_end = int(cfg["test_end_idx"])
    else:
        train_start = int(n_total * 0.01)
        train_end = int(n_total * 0.7) - 1
        test_start = train_end + 1
        test_end = n_total - 1

    if not (
        0 <= train_start <= train_end < n_total
        and 0 <= test_start <= test_end < n_total
    ):
        raise ValueError(
            "Invalid train/test ranges: "
            f"train={train_start}:{train_end}, "
            f"test={test_start}:{test_end}, "
            f"n_total={n_total}"
        )

    # zscore_normalize uses a Python slice, whose end is exclusive.
    normalized = zscore_normalize(
        raw,
        train_start,
        train_end + 1,
    )

    n_input_vars = len(input_ids)
    input_vecs = [normalized[i - 1] for i in input_ids]
    output_vec = normalized[output_id - 1]

    X_train_full, Y_train_full = make_windows_multi(
        input_vecs,
        output_vec,
        lookback,
        past_interval,
        train_start,
        train_end,
        allow_history_before_start=False,
    )

    X_test, Y_test = make_windows_multi(
        input_vecs,
        output_vec,
        lookback,
        past_interval,
        test_start,
        test_end,
        allow_history_before_start=True,
    )

    if X_train_full is None:
        print(json.dumps({"error": "not enough data"}))
        return

    X_train = X_train_full
    Y_train = Y_train_full
    n_train = len(Y_train)
    target_channel = input_ids.index(output_id)

    # Naive persistence
    pred_naive = X_test[:, -1, target_channel]

    rmse_naive, mae_naive, nrmse_naive, r2_naive = compute_metrics(
        Y_test,
        pred_naive
    )


    # ---------------------------------------------------------
    # Ridge regression
    # ---------------------------------------------------------

    X_train_r = X_train.reshape(X_train.shape[0], -1)
    X_test_r = X_test.reshape(X_test.shape[0], -1)

    ntr, d = X_train_r.shape

    X_aug = np.hstack([
        np.ones((ntr, 1)),
        X_train_r
    ])

    I_mat = np.eye(d + 1)
    I_mat[0, 0] = 0.0

    B = np.linalg.solve(
        X_aug.T @ X_aug + ridge_lambda * I_mat,
        X_aug.T @ Y_train
    )

    X_test_aug = np.hstack([
        np.ones((X_test_r.shape[0], 1)),
        X_test_r
    ])

    pred_ridge = X_test_aug @ B

    rmse_ridge, mae_ridge, nrmse_ridge, r2_ridge = compute_metrics(
        Y_test,
        pred_ridge
    )


    # ---------------------------------------------------------
    # LSTM
    # ---------------------------------------------------------

    torch.manual_seed(seed)

    X_train_t = torch.FloatTensor(X_train).permute(1, 0, 2)
    X_test_t = torch.FloatTensor(X_test).permute(1, 0, 2)
    Y_train_t = torch.FloatTensor(Y_train)

    lstm = torch.nn.LSTM(
        n_input_vars,
        hidden,
        num_layers=n_layers,
        dropout=dropout if n_layers > 1 else 0
    )

    fc = torch.nn.Linear(hidden, 1)
    drop = torch.nn.Dropout(dropout)

    optimizer = torch.optim.Adam(
        list(lstm.parameters()) + list(fc.parameters()),
        lr=lr
    )

    for epoch in range(epochs):

        lstm.train()
        fc.train()
        drop.train()

        perm = torch.randperm(n_train)

        for batch_start in range(0, n_train, batch_size):

            batch_idx = perm[
                batch_start:batch_start + batch_size
            ]

            x_batch = X_train_t[:, batch_idx, :]
            y_batch = Y_train_t[batch_idx]

            optimizer.zero_grad()

            lstm_out, _ = lstm(x_batch)

            pred = fc(
                drop(lstm_out[-1])
            ).squeeze(-1)

            loss = ((pred - y_batch) ** 2).mean()

            loss.backward()
            optimizer.step()


    # ---------------------------------------------------------
    # Test
    # ---------------------------------------------------------

    lstm.eval()
    fc.eval()
    drop.eval()

    with torch.no_grad():

        lstm_out, _ = lstm(X_test_t)

        pred_lstm = fc(
            drop(lstm_out[-1])
        ).squeeze(-1).numpy()

    rmse_lstm, mae_lstm, nrmse_lstm, r2_lstm = compute_metrics(
        Y_test,
        pred_lstm
    )

    result = {
        "mode": mode,
        "var": output_name,
        "input_ids": input_ids,
        "past_interval": past_interval,
        "lookback": lookback,
        "hidden": hidden,
        "n_layers": n_layers,
        'seed': seed,
        'ridge_lambda': ridge_lambda,
        'lstm_rmse': rmse_lstm, 'lstm_mae': mae_lstm,
        'lstm_nrmse_std': nrmse_lstm, 'lstm_r2': r2_lstm,
        'ridge_rmse': rmse_ridge, 'ridge_mae': mae_ridge,
        'ridge_nrmse_std': nrmse_ridge, 'ridge_r2': r2_ridge,
        'naive_rmse': rmse_naive, 'naive_mae': mae_naive,
        'naive_nrmse_std': nrmse_naive, 'naive_r2': r2_naive
    }

    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(
        output_dir,
        f"result_{output_name}_DS{past_interval}_L{lookback}_HU{hidden}_NL{n_layers}_S{seed}.json",
    )
    with open(result_file, 'w') as f:
        json.dump(result, f)

    print(json.dumps(result))

if __name__ == "__main__":
    main()
