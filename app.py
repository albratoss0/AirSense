# ==============================================================================
# IMPORTS
# ==============================================================================
import pandas as pd  # Data manipulation, time-series cleaning, and resampling
import requests  # HTTP client for calling the live OpenAQ REST API
import streamlit as st  # Web dashboard UI framework
from sklearn.ensemble import HistGradientBoostingRegressor  # Gradient boosting model for forecasting
from sklearn.metrics import mean_absolute_error  # Evaluation metric (average error in µg/m³)
from sklearn.model_selection import train_test_split  # Used to demonstrate why random splits leak time


# ==============================================================================
# PAGE CONFIGURATION & HEADER
# ==============================================================================
# Configure the browser tab title, full-width page layout, and cloud icon
st.set_page_config(
    page_title="Air Sense Forecasting Dashboard",
    layout="wide",
    page_icon=":cloud:",
)

# Display the main dashboard heading and an explanatory disclaimer caption
st.title("Air Sense Forecasting Dashboard")
st.caption(
    "This app works with PM2.5 concentration measurements, not a precomputed AQI. "
    "Its watch level is user-defined and is not medical guidance."
)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def prepare_series(frame, timestamp_column, value_column):
    """Clean raw CSV columns into a regular, gap-filled UTC PM2.5 time series."""
    # Keep only the user-selected timestamp and PM2.5 columns and standardize their names
    selected = frame[[timestamp_column, value_column]].copy()
    selected.columns = ["timestamp", "pm25"]

    # Parse timestamps into UTC datetimes; invalid strings become NaT (missing)
    selected["timestamp"] = pd.to_datetime(
        selected["timestamp"], errors="coerce", utc=True
    )
    # Convert PM2.5 readings to numbers; non-numeric text becomes NaN (missing)
    selected["pm25"] = pd.to_numeric(selected["pm25"], errors="coerce")

    # Drop any rows where either the timestamp or PM2.5 reading was invalid/missing
    selected = selected.dropna()

    # Average duplicate timestamps (if multiple rows share a timestamp) and sort chronologically
    series = selected.groupby("timestamp")["pm25"].mean().sort_index()
    if len(series) < 3:
        raise ValueError("The selected columns contain fewer than three usable rows.")

    # Compute the median time gap between consecutive readings to detect hourly vs. daily data
    median_gap = series.index.to_series().diff().dropna().median()
    if median_gap <= pd.Timedelta(hours=6):
        # Treat datasets with gaps <= 6 hours as hourly data: forecast next 24 hours
        cadence = "h"
        cadence_label = "hourly"
        horizon = 24
        # Lags capture recent hours (1, 2, 3), same hour yesterday (24), and same hour last week (168)
        lags = [1, 2, 3, 24, 168]
        rolling_window = 24
    else:
        # Otherwise treat as daily data: forecast next 7 days
        cadence = "D"
        cadence_label = "daily"
        horizon = 7
        # Lags capture recent days (1, 2, 3), 1 week ago (7), and 2 weeks ago (14)
        lags = [1, 2, 3, 7, 14]
        rolling_window = 7

    # Put the series onto a regular hourly/daily grid and linearly interpolate missing steps
    series = series.resample(cadence).mean().interpolate().dropna()
    series.name = "pm25"
    return series, cadence, cadence_label, horizon, lags, rolling_window


def make_features(series, lags, rolling_window):
    """Build supervised learning features (past lags + rolling average) from the time series."""
    features = pd.DataFrame(index=series.index)

    # Create lag columns (e.g., lag_1 = value 1 step ago, lag_24 = value 24 steps ago)
    for lag in lags:
        features[f"lag_{lag}"] = series.shift(lag)

    # Compute the rolling average of past values (shifted by 1 so we never peek at the current target)
    features["rolling_mean"] = series.shift(1).rolling(rolling_window).mean()

    # The value we want the model to predict at each timestamp
    features["target"] = series

    # Drop initial warm-up rows where lag or rolling window values are not yet available
    return features.dropna()


def make_model():
    """Create a reproducible Histogram-based Gradient Boosting Regressor."""
    return HistGradientBoostingRegressor(
        random_state=42,       # Fixed seed so results are repeatable
        min_samples_leaf=5,    # Prevents overfitting on tiny leaf groups
        early_stopping=False,  # Train all trees consistently across small/medium datasets
    )


def forecast_future(model, series, feature_columns, lags, rolling_window, cadence, horizon):
    """Generate multi-step future predictions recursively (feeding each prediction into the next step)."""
    history = series.copy()
    predictions = []
    timestamps = []

    # Step forward by 1 hour for hourly data, or 1 day for daily data
    step = pd.Timedelta(hours=1) if cadence == "h" else pd.Timedelta(days=1)

    for _ in range(horizon):
        next_timestamp = history.index[-1] + step

        # Build the single-row feature input for `next_timestamp` from the end of `history`
        row = {}
        for lag in lags:
            row[f"lag_{lag}"] = history.iloc[-lag]
        row["rolling_mean"] = history.iloc[-rolling_window:].mean()

        # Predict the next PM2.5 value using the trained model
        row_frame = pd.DataFrame([row], columns=feature_columns)
        prediction = float(model.predict(row_frame)[0])

        # Append prediction to `history` so subsequent future steps can use it as lag input
        history.loc[next_timestamp] = prediction
        timestamps.append(next_timestamp)
        predictions.append(prediction)

    return pd.Series(predictions, index=timestamps, name="Forecast PM2.5")


def fetch_nearby_openaq(api_key, latitude, longitude):
    """Find the closest OpenAQ PM2.5 sensor within 25 km and fetch its last 7 days of hourly readings."""
    # Authenticate with the user's OpenAQ v3 API key via request header
    headers = {"X-API-Key": api_key}

    # Step 1: Query OpenAQ /v3/locations within a 25,000m (25 km) radius for PM2.5 (parameters_id=2)
    location_response = requests.get(
        "https://api.openaq.org/v3/locations",
        headers=headers,
        params={
            "coordinates": f"{latitude:.4f},{longitude:.4f}",
            "radius": 25000,
            "parameters_id": 2,
            "limit": 100,
            "page": 1,
        },
        timeout=20,
    )
    location_response.raise_for_status()  # Raise an error if HTTP status is 4xx/5xx
    locations = location_response.json().get("results", [])

    # Collect all candidate PM2.5 sensors (parameter id == 2) along with provider & license attribution
    candidates = []
    for location in locations:
        for sensor in location.get("sensors", []):
            if sensor.get("parameter", {}).get("id") == 2:
                licenses = location.get("licenses") or []
                attribution = "See OpenAQ metadata"
                if licenses:
                    attribution = licenses[0].get("attribution", {}).get(
                        "name", attribution
                    )
                candidates.append(
                    {
                        "distance": location.get("distance"),
                        "location": location.get("name") or "Unnamed location",
                        "provider": location.get("provider", {}).get(
                            "name", "Unknown provider"
                        ),
                        "attribution": attribution,
                        "sensor_id": sensor["id"],
                    }
                )

    if not candidates:
        raise ValueError("No PM2.5 sensor was returned within 25 km.")

    # Pick the physically closest PM2.5 sensor to the given coordinates
    closest = min(
        candidates,
        key=lambda item: item["distance"]
        if item["distance"] is not None
        else float("inf"),
    )

    # Step 2: Fetch the last 7 days of hourly averages from that closest sensor
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=7)
    sensor_id = closest["sensor_id"]
    hourly_response = requests.get(
        f"https://api.openaq.org/v3/sensors/{sensor_id}/hours",
        headers=headers,
        params={
            "datetime_from": start.isoformat(),
            "datetime_to": end.isoformat(),
            "limit": 1000,
            "page": 1,
        },
        timeout=20,
    )
    hourly_response.raise_for_status()

    # Extract UTC timestamp and PM2.5 value from each returned hourly record
    rows = []
    for item in hourly_response.json().get("results", []):
        period = item.get("period") or {}
        datetime_from = period.get("datetimeFrom") or {}
        rows.append(
            {
                "timestamp": datetime_from.get("utc"),
                "value": item.get("value"),
            }
        )

    # Clean and sort the live sensor readings into a tidy DataFrame
    live = pd.DataFrame(rows)
    if live.empty:
        raise ValueError("The selected sensor returned no hourly rows for the last week.")
    live["timestamp"] = pd.to_datetime(live["timestamp"], errors="coerce", utc=True)
    live["value"] = pd.to_numeric(live["value"], errors="coerce")
    live = live.dropna().sort_values("timestamp")
    if live.empty:
        raise ValueError("The returned hourly rows had no usable timestamps and values.")

    return closest, live


# ==============================================================================
# STEP 1: CSV UPLOAD & COLUMN MAPPING
# ==============================================================================
# Prompt user to upload a CSV file; pause execution cleanly until a file is provided
uploaded_file = st.file_uploader("Upload the Pokhara CSV", type=["csv"])
if uploaded_file is None:
    st.info("Upload the Pokhara CSV to begin.")
    st.stop()

# Safely load the uploaded CSV into a pandas DataFrame
try:
    raw = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"The CSV could not be read: {exc}")
    st.stop()

# Ensure there are at least two columns (one for time, one for PM2.5)
if len(raw.columns) < 2:
    st.error("The CSV needs at least two columns.")
    st.stop()

# Let the user map which column holds timestamps and which holds PM2.5 values
timestamp_column = st.selectbox("Timestamp column", raw.columns.tolist())
value_column = st.selectbox("PM2.5 concentration column", raw.columns.tolist())
if timestamp_column == value_column:
    st.warning("Choose different timestamp and PM2.5 columns.")
    st.stop()


# ==============================================================================
# STEP 2: CLEAN SERIES, SUMMARY METRICS & RECURRING PATTERNS
# ==============================================================================
# Clean the raw data and detect whether the series is hourly or daily
try:
    series, cadence, cadence_label, horizon, lags, rolling_window = prepare_series(
        raw, timestamp_column, value_column
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

# Ensure the cleaned time series is long enough to create lags and still train/evaluate
minimum_rows = max(lags) + horizon + 20
if len(series) < minimum_rows:
    st.error(
        f"The cleaned {cadence_label} series needs at least {minimum_rows} rows "
        f"for these lags and forecast horizon. It currently has {len(series)}."
    )
    st.stop()

# Show 3 high-level KPI metrics summarizing the cleaned series
summary_1, summary_2, summary_3 = st.columns(3)
summary_1.metric("Usable observations", len(series))
summary_2.metric("Detected cadence", cadence_label)
summary_3.metric("Forecast horizon", f"{horizon} {cadence_label} steps")

# Plot the cleaned historical PM2.5 time series
st.subheader("Historical PM2.5")
st.line_chart(pd.DataFrame({"Observed PM2.5": series}))

# Plot the average daily cycle (by UTC hour) or weekly cycle (by day of week)
st.subheader("Recurring pattern")
if cadence == "h":
    pattern = series.groupby(series.index.hour).mean()
    pattern.index.name = "UTC hour"
else:
    pattern = series.groupby(series.index.day_of_week).mean()
    pattern.index.name = "Weekday number, Monday=0"
st.line_chart(pd.DataFrame({"Mean PM2.5": pattern}))


# ==============================================================================
# STEP 3: MODEL EVALUATION — RANDOM SPLIT LEAKAGE VS. HONEST CHRONOLOGICAL SPLIT
# ==============================================================================
# Build lag and rolling-mean features X and target values y
features = make_features(series, lags, rolling_window)
feature_columns = [column for column in features.columns if column != "target"]
X = features[feature_columns]
y = features["target"]

# 3A. Demonstrate why a shuffled random train/test split is misleading on time-series data:
# future timestamps end up inside X_train_random, leaking information across time
X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(
    X, y, test_size=0.2, random_state=42
)
random_model = make_model()
random_model.fit(X_train_random, y_train_random)
random_predictions = random_model.predict(X_test_random)
random_mae = mean_absolute_error(y_test_random, random_predictions)
random_mix = X_train_random.index.max() > X_test_random.index.min()

st.subheader("The tempting random split")
st.warning(f"Random split mixes time: {random_mix}")
st.metric("Random-split MAE", f"{random_mae:.2f} µg/m³")

# 3B. Perform an honest chronological split: train strictly on the past, test strictly on the latest 20%
test_size = max(horizon, int(len(X) * 0.2))
split_at = len(X) - test_size
X_train = X.iloc[:split_at]
X_test = X.iloc[split_at:]
y_train = y.iloc[:split_at]
y_test = y.iloc[split_at:]

chronological_model = make_model()
chronological_model.fit(X_train, y_train)
chronological_predictions = chronological_model.predict(X_test)
chronological_mae = mean_absolute_error(y_test, chronological_predictions)

# Benchmark against a naive "persistence" baseline (predicting that the next step equals `lag_1`)
persistence_predictions = X_test["lag_1"]
persistence_mae = mean_absolute_error(y_test, persistence_predictions)
chronology_ok = X_train.index.max() < X_test.index.min()

st.subheader("Honest future holdout")
st.success(f"Chronological split keeps train before test: {chronology_ok}")
metric_1, metric_2, metric_3 = st.columns(3)
metric_1.metric("Random MAE", f"{random_mae:.2f} µg/m³")
metric_2.metric("Chronological MAE", f"{chronological_mae:.2f} µg/m³")
metric_3.metric("Persistence MAE", f"{persistence_mae:.2f} µg/m³")


# ==============================================================================
# STEP 4: FUTURE PM2.5 FORECAST & USER-DEFINED WATCH LEVEL
# ==============================================================================
# Refit the model on 100% of available historical data before forecasting into the future
final_model = make_model()
final_model.fit(X, y)
forecast = forecast_future(
    final_model,
    series,
    feature_columns,
    lags,
    rolling_window,
    cadence,
    horizon,
)

# Plot recent observed PM2.5 alongside the recursive future forecast
st.subheader("Future PM2.5 forecast")
forecast_chart = pd.concat(
    [series.tail(horizon * 3).rename("Observed PM2.5"), forecast], axis=1
)
st.line_chart(forecast_chart)

# Default the custom watch threshold to the historical 90th percentile, editable by the user
watch_default = float(round(series.quantile(0.90), 1))
watch_level = st.number_input(
    "User-defined watch level (µg/m³)",
    min_value=0.0,
    value=watch_default,
    step=1.0,
)

# Flag each forecasted period that meets or exceeds the user's watch level
forecast_table = pd.DataFrame({"forecast_pm25": forecast})
forecast_table["at_or_above_watch_level"] = forecast_table["forecast_pm25"] >= watch_level
st.dataframe(forecast_table.round(2))

watch_count = int(forecast_table["at_or_above_watch_level"].sum())
if watch_count:
    st.warning(f"{watch_count} forecast periods meet or exceed the watch level.")
else:
    st.info("No forecast periods meet the current watch level.")


# ==============================================================================
# STEP 5: LIVE NEARBY SENSOR CONTEXT FROM OPENAQ
# ==============================================================================
st.subheader("Recent OpenAQ PM2.5 context")
st.caption("Enter the API key at runtime. It is not stored by this app.")

# Collect OpenAQ API key (masked password input) and WGS84 coordinates at runtime
api_key = st.text_input("OpenAQ API key", type="password")
latitude_text = st.text_input("Latitude from the Pokhara data or metadata")
longitude_text = st.text_input("Longitude from the Pokhara data or metadata")

# Fetch and display live nearby PM2.5 readings only when the user clicks the button
if st.button("Fetch nearby OpenAQ PM2.5"):
    if not api_key or not latitude_text or not longitude_text:
        st.error("Enter an API key, latitude, and longitude.")
    else:
        try:
            latitude = float(latitude_text)
            longitude = float(longitude_text)
            with st.spinner("Fetching recent OpenAQ measurements"):
                source, live = fetch_nearby_openaq(api_key, latitude, longitude)

            # Display sensor details and required data-provider attribution
            st.success(
                f"Loaded sensor {source['sensor_id']} at {source['location']}."
            )
            st.caption(
                f"Data source: OpenAQ. Provider: {source['provider']}. "
                f"Attribution: {source['attribution']}."
            )

            # Plot the last week of hourly readings and display the latest 24 rows in a table
            live_chart = live.set_index("timestamp")[["value"]]
            live_chart.columns = ["Recent hourly PM2.5"]
            st.line_chart(live_chart)
            st.dataframe(live.tail(24))
        except (ValueError, requests.exceptions.RequestException) as exc:
            st.error(f"OpenAQ request failed: {exc}")
