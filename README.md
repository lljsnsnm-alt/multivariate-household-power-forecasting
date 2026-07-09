# Multivariate Household Power Forecasting

This repository contains a multivariate time series forecasting project for household electric power consumption.

The task is to use the past 90 days of multivariate features to predict future global active power consumption for two forecasting horizons:

- Short-term forecasting: 90 days
- Long-term forecasting: 365 days

The project compares three models:

- LSTM
- Transformer
- RevIN-Calendar Transformer

The final improved model, RevIN-Calendar Transformer, combines weather variables, calendar periodic features, reversible instance normalization, and Transformer encoder blocks.

---

## Dataset

### 1. Household Power Consumption Data

The main electricity dataset is from the UCI Machine Learning Repository:

Individual Household Electric Power Consumption Dataset  
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

The original dataset contains minute-level measurements from a household in France from December 2006 to November 2010.

Main electricity variables include:

- `global_active_power`
- `global_reactive_power`
- `voltage`
- `global_intensity`
- `sub_metering_1`
- `sub_metering_2`
- `sub_metering_3`

### 2. Weather Data

Monthly climate data are from data.gouv.fr / Météo-France:

Données climatologiques de base mensuelles  
https://www.data.gouv.fr/fr/datasets/donnees-climatologiques-de-base-mensuelles/

The household is located in Sceaux, Hauts-de-Seine, France. Therefore, this project uses monthly weather data from department 92.

Weather variables used:

- `RR`
- `NBJRR1`
- `NBJRR5`
- `NBJRR10`
- `NBJBROU`

---

## Data Preprocessing

The original minute-level electricity data are aggregated into daily records.

Aggregation rules:

| Variable | Aggregation |
|---|---|
| `global_active_power` | daily sum |
| `global_reactive_power` | daily sum |
| `sub_metering_1` | daily sum |
| `sub_metering_2` | daily sum |
| `sub_metering_3` | daily sum |
| `voltage` | daily mean |
| `global_intensity` | daily mean |
| weather variables | monthly values merged into daily samples |

The remaining sub-metering power is calculated as:

```text
sub_metering_remainder =
global_active_power * 1000 / 60
- sub_metering_1
- sub_metering_2
- sub_metering_3
```

After preprocessing, each daily sample contains electricity features and weather features.

Final daily features include:

- `global_active_power`
- `global_reactive_power`
- `voltage`
- `global_intensity`
- `sub_metering_1`
- `sub_metering_2`
- `sub_metering_3`
- `sub_metering_remainder`
- `RR`
- `NBJRR1`
- `NBJRR5`
- `NBJRR10`
- `NBJBROU`

The input length is:

```text
90 days
```

The forecasting horizons are:

```text
90 days
365 days
```

---

## Models

### 1. LSTM

The LSTM model encodes the past 90 days of multivariate input features and uses the last hidden state to predict the future target sequence.

### 2. Transformer

The Transformer model uses linear embedding, positional encoding, and Transformer encoder layers to model long-range temporal dependencies.

### 3. RevIN-Calendar Transformer

The proposed improved model uses:

- RevIN-style sample-level normalization
- weather variables
- calendar periodic features
- positional encoding
- Transformer encoder
- attention pooling

Additional calendar features:

- `doy_sin`
- `doy_cos`
- `month_sin`
- `month_cos`

This model is designed to handle distribution shift and seasonal patterns in household power consumption.

---

## Experimental Setup

Input window:

```text
past 90 days
```

Prediction windows:

```text
future 90 days
future 365 days
```

Evaluation metrics:

- Mean Squared Error, MSE
- Mean Absolute Error, MAE

Each experiment is repeated 5 times using different random seeds. The mean and standard deviation are reported.

---

## Results

| Model | Horizon | MSE Mean | MSE Std | MAE Mean | MAE Std |
|---|---:|---:|---:|---:|---:|
| LSTM  | 90 | 217542.95 | 15697.50 | 359.49 | 16.68 |
| LSTM  | 365 | 517175.34 | 24154.54 | 603.73 | 10.98 |
| Transformer  | 90 | 238576.91 | 13841.72 | 379.19 | 14.68 |
| Transformer  | 365 | 312869.72 | 65755.86 | 444.82 | 53.74 |
| RevIN-Calendar Transformer | 90 | 184030.11 | 7004.12 | 329.50 | 6.36 |
| RevIN-Calendar Transformer | 365 | 161432.45 | 8016.35 | 310.86 | 7.57 |

The RevIN-Calendar Transformer achieves the best performance for both short-term and long-term forecasting.

---

## Project Structure

```text
.
├── README.md
├── dataset/
│   ├── daily_power.csv
│   ├── daily_power_weather.csv
│   ├── MENSQ_92_previous-1950-2024.csv.gz
│   ├── train.csv
│   └── test.csv
├── figures/
│   ├── lstm_weather_90.png
│   ├── lstm_weather_365.png
│   ├── transformer_90.png
│   ├── transformer_365.png
│   ├── revin_calendar_transformer_90.png
│   └── revin_calendar_transformer_365.png
├── my_tools/
│   ├── preprocess_power_data.py
│   ├── merge_weather.py
│   └── test_windows.py
├── train_lstm.py
├── train_transformer.py
├── train_revin_calendar_transformer.py
├── lstm_weather_results.csv
├── transformer_results.csv
└── revin_calendar_transformer_results.csv
```

---

## How to Run

### 1. Install Dependencies

```bash
pip install numpy pandas matplotlib torch
```

Or use conda:

```bash
conda install numpy pandas matplotlib -y
conda install pytorch torchvision torchaudio -c pytorch -y
```

### 2. Preprocess Electricity Data

```bash
python my_tools/preprocess_power_data.py
```

This step converts the original minute-level power data into daily-level data.

### 3. Merge Weather Data

```bash
python my_tools/merge_weather.py
```

This step merges monthly weather variables from department 92 into the daily power dataset.

### 4. Train LSTM

```bash
python train_lstm.py
```

### 5. Train Transformer

```bash
python train_transformer.py
```

### 6. Train RevIN-Calendar Transformer

```bash
python train_revin_calendar_transformer.py
```

If using a specific GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python train_revin_calendar_transformer.py
```

---

## Output Files

Training produces result CSV files and prediction figures.

Result files:

```text
lstm_weather_results.csv
transformer_results.csv
revin_calendar_transformer_results.csv
```

Prediction figures are saved in:

```text
figures/
```

---

## Notes

The original raw electricity data file is not included in this repository because it may be large. It can be downloaded from the official UCI dataset link.

The processed files included in `dataset/` are used for model training and testing:

- `daily_power.csv`
- `daily_power_weather.csv`
- `train.csv`
- `test.csv`

The weather file used in this project is:

```text
MENSQ_92_previous-1950-2024.csv.gz
```

---

## References

[1] UCI Machine Learning Repository. Individual household electric power consumption dataset.  
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

[2] data.gouv.fr. Données climatologiques de base mensuelles, Météo-France.  
https://www.data.gouv.fr/fr/datasets/donnees-climatologiques-de-base-mensuelles/

[3] Hochreiter, S., & Schmidhuber, J. Long Short-Term Memory. Neural Computation, 1997.

[4] Vaswani, A., et al. Attention Is All You Need. NeurIPS, 2017.

[5] Lim, B., & Zohren, S. Time-series forecasting with deep learning: a survey. Philosophical Transactions of the Royal Society A, 2021.

[6] Kim, T., Kim, J., Tae, Y., Park, C., Choi, J. H., & Choo, J. Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift. ICLR, 2022.

---

## Acknowledgement

This project documentation was partially assisted by ChatGPT for language polishing and organization. The data preprocessing, model training, result recording, and analysis were completed by the author.
```
