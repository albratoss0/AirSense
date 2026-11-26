# Air Sense Forecasting Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://airsense-rwp3mkrebuno3vaxuentfg.streamlit.app/)

I built and deployed **Air Sense**, an interactive time-series forecasting web app in Python and Streamlit. It comes preloaded with 5,500+ air quality records across 15 cities—and also lets users upload their own sensor CSVs—to clean messy readings, uncover weekly and seasonal pollution patterns, forecast future levels using gradient boosting, and pull live nearby sensor data from the OpenAQ API. My favorite part is that it visually proves why random train/test splits leak future data in time-series ML and benchmarks the model honestly against a naive persistence baseline.

- **Live Demo:** [https://airsense-rwp3mkrebuno3vaxuentfg.streamlit.app/](https://airsense-rwp3mkrebuno3vaxuentfg.streamlit.app/)
- **Source Code:** [https://github.com/albratoss0/AirSense](https://github.com/albratoss0/AirSense)

---

## How It Works Under the Hood

### 1. Ingestion & Automated Cleaning (`prepare_series`)
- Raw environmental data is messy: missing values, duplicate timestamps, mixed cities, and sensor spikes (like single-day 460+ µg/m³ glitches).
- The pipeline auto-detects timestamp, location, and numeric columns, parses timestamps to UTC, clips extreme anomalies with a `3.5x IQR` filter, detects whether the data is **hourly or daily** from the median time gap, and resamples onto a uniform grid with time interpolation.

### 2. Feature Engineering (`make_features`)
- Since tree models don't natively understand time order, I convert the series into supervised tabular features: **lagged values** (e.g., 1, 2, 3, 7, and 14 steps back), **shifted rolling mean and standard deviation** (shifted by 1 step so the current target never leaks into its own features), and **calendar features** (day of week, hour/month).

### 3. Honest Validation (`Chronological` vs. `Random` vs. `Persistence`)
- I train a regularized `HistGradientBoostingRegressor` in `scikit-learn` and evaluate Mean Absolute Error (MAE) on the **final 20% chronological holdout**, comparing it side by side against both a **leaky shuffled split** and a **lag-1 persistence baseline** (predicting tomorrow equals today).

### 4. Recursive Forecasting & Live Context (`forecast_future` & `fetch_nearby_openaq`)
- After validation, the model retrains on 100% of historical data and generates multi-step forecasts **recursively** (feeding each predicted step back into the lag history for the next step), flags periods above a user-defined watch threshold, and queries the **OpenAQ v3 REST API** at runtime for live readings within 25 km without storing API keys.

---

## Tech Stack
- **Language:** Python
- **Machine Learning:** `scikit-learn` (`HistGradientBoostingRegressor`)
- **Data Processing:** `pandas`
- **Dashboard UI:** `Streamlit`
- **Live Sensor Data:** `requests` + OpenAQ v3 REST API

---

## Run Locally

```bash
git clone https://github.com/albratoss0/AirSense.git
cd AirSense
pip install -r requirements.txt
streamlit run app.py
```
