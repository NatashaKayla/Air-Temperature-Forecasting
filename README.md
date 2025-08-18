# 🌤️ Air Temperature Forecasting — Time-Series EDA & LSTM

End-to-end notebook to **forecast air temperature** from multivariate weather readings.
Pipeline: **preprocessing → EDA & feature selection → scaling & windowing → LSTM (baseline & modified) → evaluation**.

---

## 🎯 Objectives

* Understand temporal patterns & relationships among variables
* Build a clean, leak-free supervised dataset from a time series
* Compare **baseline vs. modified LSTM** and report robust metrics (**MAE**, **MSE**, **R²**, **MAPE**)

---

## 🗂️ Dataset

* Time-indexed weather observations with air temperature as the **target**.
* Multiple predictors (see notebook for the full column list), including **`from date`** (timestamp) and **`VWS`**.

---

## 🧼 Data Preprocessing

* **Duplicate check & removal** to ensure unique timestamps.
* **Datetime parsing:** convert **`from date`** to proper `datetime`, then **sort** chronologically.
* **Missing values:** impute nulls (per-column strategies documented in the notebook).
* **Interpolation:** apply **linear interpolation** on **`VWS`** to fill small gaps.
* **Capping:** cap extreme outliers to sensible bounds (e.g., percentile-based).
* **Feature selection:** retain the most informative predictors based on EDA & correlations.

---

## 🔎 EDA Highlights

* **Distributions** of key variables & the target
* **Seasonality check** (trend/periodicity across time)
* **Correlation heatmap** among numeric features
* Notes from **feature selection** (kept vs. dropped) included in the notebook

---

## 🔧 Scaling & Windowing

* **Chronological split:** **train → validation → test** (prevents leakage).
* **MinMaxScaler** fitted on **train only**, applied to val/test.
* **Sliding windows:** past `input_width` steps → predict next step (`label_width`/`shift` configurable).

---

## 🧠 Modeling

* **Baseline LSTM:** compact single-stack LSTM + dense head.
* **Modified LSTM:** deeper/wider LSTM with tuned units, dropout, and training hyperparameters.
* **Training:** loss **MSE**, optimizer **Adam**, with **EarlyStopping** & **ModelCheckpoint**.

---

## 📈 Evaluation

* Reported on validation/test:

  * **MAE**, **MSE**, **R²**, **MAPE**
* Forecast vs. actual plots for visual inspection (see notebook).
* **Summary:** The **modified LSTM** outperforms the baseline across all metrics (lower MAE/MSE/MAPE, higher R²).

---

## 📌 Conclusion

* Data integrity ensured: duplicates removed, **`from date`** parsed & sorted, missing values handled, **VWS** interpolated, and outliers **capped**.
* EDA (distributions, seasonality, correlations) informed **feature selection**, yielding a compact, predictive set.
* Time-aware split + **MinMax** scaling (fit on train only) and **sliding-window** framing prevented leakage.
* **Modified LSTM** beat the baseline on **MAE, MSE, R², and MAPE**—more accurate, better generalization, lower absolute and percentage errors.
* Pipeline is reproducible and ready for extensions (tuning, alternative architectures, probabilistic intervals, deployment).
