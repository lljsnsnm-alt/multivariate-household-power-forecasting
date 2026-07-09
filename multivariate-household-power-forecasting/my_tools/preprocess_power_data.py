import pandas as pd
import numpy as np

path = r"/data/bfx/code/new_project/dataset/household_power_consumption.txt"

df = pd.read_csv(path, sep=";", na_values="?", low_memory=False)

df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H:%M:%S"
)

df = df.drop(columns=["Date", "Time"]).set_index("datetime")
df = df.astype(float)
df = df.interpolate(method="time").ffill().bfill()

daily = pd.DataFrame()

daily["global_active_power"] = df["Global_active_power"].resample("D").sum()
daily["global_reactive_power"] = df["Global_reactive_power"].resample("D").sum()
daily["voltage"] = df["Voltage"].resample("D").mean()
daily["global_intensity"] = df["Global_intensity"].resample("D").mean()
daily["sub_metering_1"] = df["Sub_metering_1"].resample("D").sum()
daily["sub_metering_2"] = df["Sub_metering_2"].resample("D").sum()
daily["sub_metering_3"] = df["Sub_metering_3"].resample("D").sum()

daily["sub_metering_remainder"] = (
    daily["global_active_power"] * 1000 / 60
    - daily["sub_metering_1"]
    - daily["sub_metering_2"]
    - daily["sub_metering_3"]
)

daily = daily.interpolate().ffill().bfill()

split_idx = int(len(daily) * 0.8)

train = daily.iloc[:split_idx]
test = daily.iloc[split_idx:]

daily.to_csv("daily_power.csv")
train.to_csv("train.csv")
test.to_csv("test.csv")

print(daily.shape)
print(train.shape)
print(test.shape)