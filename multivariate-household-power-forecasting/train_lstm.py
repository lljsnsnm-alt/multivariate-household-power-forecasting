import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = "/data/bfx/code/new_project/dataset/daily_power_weather.csv"
TARGET_COL = "global_active_power"

INPUT_LEN = 90
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
SEEDS = [2024, 2025, 2026, 2027, 2028]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_windows(data, target_col, input_len, output_len):
    X, y = [], []
    values = data.values
    target = data[target_col].values

    for i in range(len(data) - input_len - output_len + 1):
        X.append(values[i:i + input_len])
        y.append(target[i + input_len:i + input_len + output_len])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def normalize_data(X_train, y_train, X_test, y_test):
    x_mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    x_std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    x_std[x_std < 1e-6] = 1.0

    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std < 1e-6:
        y_std = 1.0

    X_train = np.clip((X_train - x_mean) / x_std, -5, 5)
    X_test = np.clip((X_test - x_mean) / x_std, -5, 5)
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    return X_train, y_train, X_test, y_test, {"y_mean": y_mean, "y_std": y_std}


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_len, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_len),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_one_run(X_train, y_train, X_test, y_test, output_len, seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train, X_test, y_test, stats = normalize_data(
        X_train, y_train, X_test, y_test
    )

    loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = LSTMForecaster(
        input_dim=X_train.shape[-1],
        hidden_dim=64,
        num_layers=2,
        output_len=output_len,
        dropout=0.2,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            pred = model(bx)
            loss = criterion(pred, by)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * bx.size(0)

        total_loss /= len(loader.dataset)

        if epoch == 1 or epoch % 20 == 0:
            print(f"Seed {seed} | Horizon {output_len} | Epoch {epoch:03d}/{EPOCHS} | Loss {total_loss:.6f}")

    model.eval()
    with torch.no_grad():
        pred_norm = model(torch.tensor(X_test).to(device)).cpu().numpy()

    pred = pred_norm * stats["y_std"] + stats["y_mean"]
    true = y_test * stats["y_std"] + stats["y_mean"]

    mse = np.mean((pred - true) ** 2)
    mae = np.mean(np.abs(pred - true))
    return mse, mae, pred, true


def plot_prediction(pred, true, output_len, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(true[0], label="Ground Truth", linewidth=2)
    plt.plot(pred[0], label="Prediction", linewidth=2)
    plt.title(f"LSTM Weather Power Forecast ({output_len} days)")
    plt.xlabel("Day")
    plt.ylabel("Global Active Power")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_experiment(daily, output_len):
    test_days = INPUT_LEN + 365
    train_df = daily.iloc[:-test_days].copy()
    test_df = daily.iloc[-test_days:].copy()

    X_train, y_train = create_windows(train_df, TARGET_COL, INPUT_LEN, output_len)
    X_test, y_test = create_windows(test_df, TARGET_COL, INPUT_LEN, output_len)

    print("=" * 60)
    print(f"LSTM experiment: {INPUT_LEN} -> {output_len}")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    mses, maes = [], []
    best_mse, best_pred, best_true = None, None, None

    for seed in SEEDS:
        mse, mae, pred, true = train_one_run(X_train, y_train, X_test, y_test, output_len, seed)
        mses.append(mse)
        maes.append(mae)
        print(f"Seed {seed} | Horizon {output_len} | MSE: {mse:.6f} | MAE: {mae:.6f}")

        if best_mse is None or mse < best_mse:
            best_mse, best_pred, best_true = mse, pred, true

    result = {
        "model": "LSTM",
        "horizon": output_len,
        "mse_mean": np.mean(mses),
        "mse_std": np.std(mses),
        "mae_mean": np.mean(maes),
        "mae_std": np.std(maes),
    }

    print("-" * 60)
    print(result)

    plot_prediction(best_pred, best_true, output_len, f"figures/lstm_weather_{output_len}.png")
    return result


def main():
    daily = pd.read_csv(DATA_PATH, index_col="datetime")
    results = pd.DataFrame([
        run_experiment(daily, 90),
        run_experiment(daily, 365),
    ])
    results.to_csv("lstm_weather_results.csv", index=False)
    print(results)


if __name__ == "__main__":
    main()