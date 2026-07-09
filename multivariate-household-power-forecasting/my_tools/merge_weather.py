import numpy as np
import pandas as pd


POWER_PATH = "/data/bfx/code/new_project/dataset/daily_power.csv"
WEATHER_PATH = "/data/bfx/code/new_project/dataset/MENSQ_92_previous-1950-2024.csv.gz"
OUT_PATH = "/data/bfx/code/new_project/dataset/daily_power_weather.csv"

WEATHER_COLS = ["RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]


def main():
    daily = pd.read_csv(POWER_PATH)
    daily["datetime"] = pd.to_datetime(daily["datetime"])
    daily["month"] = daily["datetime"].dt.to_period("M").astype(str)

    weather = pd.read_csv(
        WEATHER_PATH,
        sep=";",
        compression="gzip",
        low_memory=False,
    )

    print("weather columns:")
    print(weather.columns.tolist())

    # 月份列通常叫 AAAAMM，例如 200612
    weather["month"] = pd.to_datetime(
        weather["AAAAMM"].astype(str),
        format="%Y%m",
        errors="coerce",
    ).dt.to_period("M").astype(str)

    for col in WEATHER_COLS:
        if col in weather.columns:
            weather[col] = pd.to_numeric(weather[col], errors="coerce")

    # 如果一个省份里有多个气象站，先选数据最完整的站
    if "NUM_POSTE" in weather.columns:
        station_scores = (
            weather.groupby("NUM_POSTE")[WEATHER_COLS]
            .count()
            .sum(axis=1)
            .sort_values(ascending=False)
        )
        best_station = station_scores.index[0]
        print("selected station:", best_station)
        weather = weather[weather["NUM_POSTE"] == best_station].copy()

    weather_monthly = weather[["month"] + WEATHER_COLS].copy()
    weather_monthly = weather_monthly.drop_duplicates("month")

    # 题目说明 RR 是毫米的十分之一，所以除以 10
    if "RR" in weather_monthly.columns:
        weather_monthly["RR"] = weather_monthly["RR"] / 10.0

    merged = daily.merge(weather_monthly, on="month", how="left")

    for col in WEATHER_COLS:
        merged[col] = merged[col].interpolate().ffill().bfill()

    merged = merged.drop(columns=["month"])
    merged.to_csv(OUT_PATH, index=False)

    print("saved:", OUT_PATH)
    print("shape:", merged.shape)
    print(merged.head())


if __name__ == "__main__":
    main()