import math
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


def create_windows(data, target_col, input_len=90, output_len=90):
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
    x_std[x_std == 0] = 1

    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std == 0:
        y_std = 1

    X_train_norm = (X_train - x_mean) / x_std
    X_test_norm = (X_test - x_mean) / x_std

    y_train_norm = (y_train - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std

    stats = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }

    return X_train_norm, y_train_norm, X_test_norm, y_test_norm, stats


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        output_len,
        dropout=0.2,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=INPUT_LEN)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_len),
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        encoded = self.encoder(x)

        last_token = encoded[:, -1, :]
        pred = self.regressor(last_token)
        return pred


def train_one_run(X_train, y_train, X_test, y_test, output_len, seed):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train, X_test, y_test, stats = normalize_data(
        X_train, y_train, X_test, y_test
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    model = TransformerForecaster(
        input_dim=X_train.shape[-1],
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        output_len=output_len,
        dropout=0.2,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_x)
            loss = criterion(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= len(train_loader.dataset)

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Seed {seed} | Horizon {output_len} | "
                f"Epoch {epoch:03d}/{EPOCHS} | Loss {epoch_loss:.6f}"
            )

    model.eval()
    with torch.no_grad():
        test_x = torch.tensor(X_test, dtype=torch.float32).to(device)
        pred_norm = model(test_x).cpu().numpy()

    pred = pred_norm * stats["y_std"] + stats["y_mean"]
    true = y_test * stats["y_std"] + stats["y_mean"]

    mse = np.mean((pred - true) ** 2)
    mae = np.mean(np.abs(pred - true))

    return mse, mae, pred, true


def plot_prediction(pred, true, output_len, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    pred_curve = pred[0]
    true_curve = true[0]

    plt.figure(figsize=(12, 5))
    plt.plot(true_curve, label="Ground Truth", linewidth=2)
    plt.plot(pred_curve, label="Prediction", linewidth=2)
    plt.title(f"Transformer Power Forecast ({output_len} days)")
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

    X_train, y_train = create_windows(
        train_df,
        target_col=TARGET_COL,
        input_len=INPUT_LEN,
        output_len=output_len,
    )

    X_test, y_test = create_windows(
        test_df,
        target_col=TARGET_COL,
        input_len=INPUT_LEN,
        output_len=output_len,
    )

    print("=" * 60)
    print(f"Transformer experiment: {INPUT_LEN} -> {output_len}")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    mses = []
    maes = []

    best_mse = None
    best_pred = None
    best_true = None

    for seed in SEEDS:
        mse, mae, pred, true = train_one_run(
            X_train,
            y_train,
            X_test,
            y_test,
            output_len=output_len,
            seed=seed,
        )

        mses.append(mse)
        maes.append(mae)

        print(
            f"Seed {seed} | Horizon {output_len} | "
            f"MSE: {mse:.6f} | MAE: {mae:.6f}"
        )

        if best_mse is None or mse < best_mse:
            best_mse = mse
            best_pred = pred
            best_true = true

    mse_mean = np.mean(mses)
    mse_std = np.std(mses)
    mae_mean = np.mean(maes)
    mae_std = np.std(maes)

    print("-" * 60)
    print(f"Transformer {output_len}-day result")
    print(f"MSE mean: {mse_mean:.6f}")
    print(f"MSE std : {mse_std:.6f}")
    print(f"MAE mean: {mae_mean:.6f}")
    print(f"MAE std : {mae_std:.6f}")

    plot_prediction(
        best_pred,
        best_true,
        output_len,
        save_path=f"figures/transformer_{output_len}.png",
    )

    return {
        "model": "Transformer",
        "horizon": output_len,
        "mse_mean": mse_mean,
        "mse_std": mse_std,
        "mae_mean": mae_mean,
        "mae_std": mae_std,
    }


def main():
    daily = pd.read_csv(DATA_PATH, index_col="datetime")

    result_90 = run_experiment(daily, output_len=90)
    result_365 = run_experiment(daily, output_len=365)

    results = pd.DataFrame([result_90, result_365])
    results.to_csv("transformer_results.csv", index=False)

    print("=" * 60)
    print("Final results:")
    print(results)
    print("Saved results to transformer_results.csv")
    print("Saved figures to figures/transformer_90.png and figures/transformer_365.png")


if __name__ == "__main__":
    main()