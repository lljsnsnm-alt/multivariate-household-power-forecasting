import math
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# DATA_PATH = "daily_power.csv"
# # 如果想用天气数据，改成：
DATA_PATH = "/data/bfx/code/new_project/dataset/daily_power_weather.csv"

TARGET_COL = "global_active_power"

INPUT_LEN = 90
BATCH_SIZE = 32
EPOCHS = 120
LR = 5e-4

SEEDS = [2024, 2025, 2026, 2027, 2028]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_calendar_features(df):
    df = df.copy()
    dates = pd.to_datetime(df.index)

    dayofyear = dates.dayofyear.values
    month = dates.month.values

    df["doy_sin"] = np.sin(2 * np.pi * dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * dayofyear / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    return df


def create_windows(data, target_col, input_len=90, output_len=90):
    X, y = [], []

    values = data.values
    target = data[target_col].values
    target_idx = list(data.columns).index(target_col)

    for i in range(len(data) - input_len - output_len + 1):
        X.append(values[i:i + input_len])
        y.append(target[i + input_len:i + input_len + output_len])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, target_idx


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


class RevINCalendarTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        output_len,
        dropout=0.15,
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=INPUT_LEN,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_len),
        )

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        std = torch.clamp(std, min=1e-5)

        x = (x - mean) / std
        x = torch.clamp(x, -5, 5)

        x = self.input_projection(x)
        x = self.positional_encoding(x)
        encoded = self.encoder(x)

        last_token = encoded[:, -1, :]
        mean_token = encoded.mean(dim=1)

        attn_score = self.attn_pool(encoded)
        attn_weight = torch.softmax(attn_score, dim=1)
        attn_token = torch.sum(encoded * attn_weight, dim=1)

        features = torch.cat([last_token, mean_token, attn_token], dim=1)
        pred_norm = self.regressor(features)

        return pred_norm


def make_target_norm(batch_x, batch_y, target_idx):
    target_history = batch_x[:, :, target_idx]
    target_mean = target_history.mean(dim=1, keepdim=True)
    target_std = target_history.std(dim=1, keepdim=True)
    target_std = torch.clamp(target_std, min=1e-5)

    batch_y_norm = (batch_y - target_mean) / target_std
    return batch_y_norm, target_mean, target_std


def train_one_run(X_train, y_train, X_test, y_test, target_idx, output_len, seed):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model = RevINCalendarTransformer(
        input_dim=X_train.shape[-1],
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        output_len=output_len,
        dropout=0.15,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-5,
    )

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            batch_y_norm, _, _ = make_target_norm(batch_x, batch_y, target_idx)

            pred_norm = model(batch_x)
            loss = criterion(pred_norm, batch_y_norm)

            if not torch.isfinite(loss):
                print(f"Seed {seed} | Horizon {output_len} | Non-finite loss detected.")
                return np.nan, np.nan, None, None

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= len(train_loader.dataset)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"Seed {seed} | Horizon {output_len} | "
                f"Epoch {epoch:03d}/{EPOCHS} | Loss {epoch_loss:.6f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_x = torch.tensor(X_test, dtype=torch.float32).to(device)
        pred_norm = model(test_x)

        target_history = test_x[:, :, target_idx]
        target_mean = target_history.mean(dim=1, keepdim=True)
        target_std = target_history.std(dim=1, keepdim=True)
        target_std = torch.clamp(target_std, min=1e-5)

        pred = pred_norm * target_std + target_mean
        pred = pred.cpu().numpy()

    true = y_test

    mse = np.mean((pred - true) ** 2)
    mae = np.mean(np.abs(pred - true))

    return mse, mae, pred, true


def plot_prediction(pred, true, output_len, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(true[0], label="Ground Truth", linewidth=2)
    plt.plot(pred[0], label="Prediction", linewidth=2)
    plt.title(f"RevIN-Calendar Transformer Forecast ({output_len} days)")
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

    X_train, y_train, target_idx = create_windows(
        train_df,
        target_col=TARGET_COL,
        input_len=INPUT_LEN,
        output_len=output_len,
    )

    X_test, y_test, _ = create_windows(
        test_df,
        target_col=TARGET_COL,
        input_len=INPUT_LEN,
        output_len=output_len,
    )

    print("=" * 60)
    print(f"RevIN-Calendar Transformer experiment: {INPUT_LEN} -> {output_len}")
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
            target_idx=target_idx,
            output_len=output_len,
            seed=seed,
        )

        print(
            f"Seed {seed} | Horizon {output_len} | "
            f"MSE: {mse:.6f} | MAE: {mae:.6f}"
        )

        if np.isfinite(mse) and np.isfinite(mae):
            mses.append(mse)
            maes.append(mae)

            if best_mse is None or mse < best_mse:
                best_mse = mse
                best_pred = pred
                best_true = true

    if len(mses) == 0:
        raise RuntimeError(f"All runs failed for horizon {output_len}.")

    result = {
        "model": "RevIN-Calendar-Transformer",
        "horizon": output_len,
        "mse_mean": np.mean(mses),
        "mse_std": np.std(mses),
        "mae_mean": np.mean(maes),
        "mae_std": np.std(maes),
        "valid_runs": len(mses),
    }

    print("-" * 60)
    print(f"RevIN-Calendar Transformer {output_len}-day result")
    print(f"Valid runs: {result['valid_runs']}/{len(SEEDS)}")
    print(f"MSE mean: {result['mse_mean']:.6f}")
    print(f"MSE std : {result['mse_std']:.6f}")
    print(f"MAE mean: {result['mae_mean']:.6f}")
    print(f"MAE std : {result['mae_std']:.6f}")

    plot_prediction(
        best_pred,
        best_true,
        output_len,
        save_path=f"figures/revin_calendar_transformer_{output_len}.png",
    )

    return result


def main():
    daily = pd.read_csv(DATA_PATH, index_col="datetime")
    daily = add_calendar_features(daily)

    result_90 = run_experiment(daily, output_len=90)
    result_365 = run_experiment(daily, output_len=365)

    results = pd.DataFrame([result_90, result_365])
    results.to_csv("revin_calendar_transformer_results.csv", index=False)

    print("=" * 60)
    print("Final results:")
    print(results)
    print("Saved results to revin_calendar_transformer_results.csv")
    print(
        "Saved figures to figures/revin_calendar_transformer_90.png "
        "and figures/revin_calendar_transformer_365.png"
    )


if __name__ == "__main__":
    main()