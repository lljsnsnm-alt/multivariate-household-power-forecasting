import numpy as np
import pandas as pd


def create_windows(data, target_col, input_len=90, output_len=90):
    X, y = [], []

    values = data.values
    target = data[target_col].values

    for i in range(len(data) - input_len - output_len + 1):
        X.append(values[i:i + input_len])
        y.append(target[i + input_len:i + input_len + output_len])

    return np.array(X), np.array(y)


daily = pd.read_csv("/data/bfx/code/new_project/dataset/daily_power.csv", index_col="datetime")

test_days = 90 + 365
train_df = daily.iloc[:-test_days]
test_df = daily.iloc[-test_days:]

X_train_90, y_train_90 = create_windows(
    train_df,
    target_col="global_active_power",
    input_len=90,
    output_len=90
)

X_test_90, y_test_90 = create_windows(
    test_df,
    target_col="global_active_power",
    input_len=90,
    output_len=90
)

X_train_365, y_train_365 = create_windows(
    train_df,
    target_col="global_active_power",
    input_len=90,
    output_len=365
)

X_test_365, y_test_365 = create_windows(
    test_df,
    target_col="global_active_power",
    input_len=90,
    output_len=365
)

print("90天短期预测：")
print("X_train_90:", X_train_90.shape)
print("y_train_90:", y_train_90.shape)
print("X_test_90:", X_test_90.shape)
print("y_test_90:", y_test_90.shape)

print("\n365天长期预测：")
print("X_train_365:", X_train_365.shape)
print("y_train_365:", y_train_365.shape)
print("X_test_365:", X_test_365.shape)
print("y_test_365:", y_test_365.shape)