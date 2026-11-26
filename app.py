# ==============================================================================
# IMPORTS
# ==============================================================================
from pathlib import Path  # Cross-platform file path handling for the built-in CSV

import pandas as pd  # Data loading, cleaning, time-series resampling, and tables
import requests  # HTTP client for live nearby sensor lookups via the OpenAQ API
import streamlit as st  # Interactive web dashboard framework
from sklearn.ensemble import HistGradientBoostingRegressor  # Gradient boosting model for forecasting
from sklearn.metrics import mean_absolute_error  # Measures average prediction error
from sklearn.model_selection import train_test_split  # Demonstrates why random splits leak future data


# ==============================================================================
# PAGE CONFIGURATION & HEADER
# ==============================================================================
# Set browser tab title, wide responsive layout, and cloud icon
st.set_page_config(
    page_title="Air Sense Forecasting Dashboard",
    layout="wide",
    page_icon=":cloud:",
)

# Main dashboard title and universal description (works for any city or air quality index/pollutant)
st.title("Air Sense Forecasting Dashboard")
st.caption(
    "Universal air quality & environmental forecasting platform. Explore the built-in multi-city dataset "
    "or upload your own CSV to inspect cleaned history, recurring patterns, honest model validation, "
    "future forecasts, custom watch-level alerts, and live OpenAQ sensor readings."
)


# ==============================================================================
# CONSTANTS & REFERENCE METADATA
# ==============================================================================
# Path to the predefined dataset bundled alongside app.py
DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "default_air_quality.csv"

# Standard measurement units for common air quality and weather columns
METRIC_UNITS = {
    "AQI": "AQI",
    "PM2.5": "µg/m³",
    "PM10": "µg/m³",
    "Ozone": "ppb",
    "NO2": "ppb",
    "CO": "ppm",
    "SO2": "ppb",
    "Temperature": "°C",
    "Humidity": "%",
}

# Predefined WGS84 coordinates (latitude, longitude) for cities in the built-in dataset
# so the OpenAQ live sensor lookup auto-fills coordinates convenience-first
CITY_COORDINATES = {
    "Austin": (30.2672, -97.7431),
    "Charlotte": (35.2271, -80.8431),
    "Chicago": (41.8781, -87.6298),
    "Columbus": (39.9612, -82.9988),
    "Dallas": (32.7767, -96.7970),
    "Fort Worth": (32.7555, -97.3308),
    "Houston": (29.7604, -95.3698),
    "Jacksonville": (30.3322, -81.6557),
    "Los Angeles": (34.0522, -118.2437),
    "New York": (40.7128, -74.0060),
    "Philadelphia": (39.9526, -75.1652),
    "Phoenix": (33.4484, -112.0740),
    "San Antonio": (29.4241, -98.4936),
    "San Diego": (32.7157, -117.1611),
    "San Jose": (37.3382, -121.8863),
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_builtin_dataset(path_str):
    """Load and cache the predefined CSV dataset from disk."""
    return pd.read_csv(path_str)


def detect_columns(frame):
    """Automatically identify timestamp, location/category, and numeric value columns."""
    columns = frame.columns.tolist()

    # 1. Detect the best timestamp column by common names or datetime parse rate
    timestamp_candidates = []
    for col in columns:
        col_lower = str(col).strip().lower()
        if any(key in col_lower for key in ("date", "time", "timestamp", "datetime")):
            timestamp_candidates.append(col)

    if not timestamp_candidates:
        for col in columns:
            if pd.api.types.is_numeric_dtype(frame[col]):
                continue
            sample = frame[col].dropna().astype(str).head(50)
            if len(sample) > 0:
                parsed = pd.to_datetime(sample, errors="coerce", utc=True)
                if parsed.notna().mean() >= 0.7:
                    timestamp_candidates.append(col)

    default_timestamp = timestamp_candidates[0] if timestamp_candidates else columns[0]

    # 2. Detect numeric-capable columns (excluding the timestamp column)
    numeric_columns = []
    for col in columns:
        if col == default_timestamp:
            continue
        converted = pd.to_numeric(frame[col], errors="coerce")
        if converted.notna().sum() >= 5 and converted.notna().mean() >= 0.3:
            numeric_columns.append(col)

    # 3. Detect optional location/grouping columns (e.g. City, Station, Location, Country)
    group_columns = []
    for col in columns:
        if col == default_timestamp or col in numeric_columns:
            continue
        nunique = frame[col].dropna().nunique()
        if 1 < nunique <= 200:
            group_columns.append(col)

    # Prioritize common location names first if multiple categorical columns exist
    group_columns.sort(
        key=lambda c: 0
        if any(k in str(c).lower() for k in ("city", "location", "station", "site", "region", "country"))
        else 1
    )

    return default_timestamp, numeric_columns, group_columns


def get_unit_for_column(column_name):
    """Return a clean display unit for known metrics, or an empty string for custom columns."""
    if column_name in METRIC_UNITS:
        return METRIC_UNITS[column_name]
    lower = str(column_name).lower()
    if "pm2.5" in lower or "pm25" in lower or "pm10" in lower:
        return "µg/m³"
    if "aqi" in lower:
        return "AQI"
    if "temp" in lower:
        return "°C"
    if "humid" in lower:
        return "%"
    return "units"


def prepare_series(frame, timestamp_column, value_column, clip_outliers=True, custom_horizon=None):
    """Clean raw rows into a gap-filled time series and configure cadence-appropriate lags."""
    # Extract only the timestamp and target metric columns
    selected = frame[[timestamp_column, value_column]].copy()
    selected.columns = ["timestamp", "value"]

    # Convert timestamps to UTC and target values to numeric floats (invalid entries become NaN)
    selected["timestamp"] = pd.to_datetime(selected["timestamp"], errors="coerce", utc=True)
    selected["value"] = pd.to_numeric(selected["value"], errors="coerce")
    raw_valid_rows = int(selected["timestamp"].notna().sum())
    missing_values = int(selected["value"].isna().sum())

    # Remove rows missing either a valid timestamp or numeric measurement
    selected = selected.dropna()
    if len(selected) < 8:
        raise ValueError(
            f"The selected columns only contain {len(selected)} valid numeric rows. "
            "At least 8 valid observations are required to build a forecast."
        )

    # Average duplicate timestamps (e.g. multiple readings on the same date/hour) and sort chronologically
    series = selected.groupby("timestamp")["value"].mean().sort_index()
    if len(series) < 8:
        raise ValueError(
            f"After grouping duplicate timestamps, only {len(series)} unique time points remain. "
            "Choose a more granular filter or upload a longer history."
        )

    # Optionally clip extreme sensor malfunction spikes using a generous 3.5x IQR fence
    outliers_clipped = 0
    if clip_outliers and len(series) >= 12:
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        if iqr > 0:
            lower_fence = q1 - 3.5 * iqr
            upper_fence = q3 + 3.5 * iqr
            outlier_mask = (series < lower_fence) | (series > upper_fence)
            outliers_clipped = int(outlier_mask.sum())
            series = series.clip(lower=lower_fence, upper=upper_fence)

    # Detect whether readings are hourly (median gap <= 6h) or daily
    median_gap = series.index.to_series().diff().dropna().median()
    if pd.notna(median_gap) and median_gap <= pd.Timedelta(hours=6):
        cadence = "h"
        cadence_label = "hourly"
        default_horizon = 24
        preferred_lags = [1, 2, 3, 24, 168]
        preferred_window = 24
    else:
        cadence = "D"
        cadence_label = "daily"
        default_horizon = 14
        preferred_lags = [1, 2, 3, 7, 14]
        preferred_window = 7

    # Resample onto a uniform hourly or daily grid and linearly fill missing gaps
    series = series.resample(cadence).mean().interpolate(method="time").bfill().ffill().dropna()
    series.name = value_column

    horizon = int(custom_horizon) if custom_horizon is not None else default_horizon

    # Dynamically adapt lags & rolling window if a user uploads a shorter custom dataset
    # so the app never crashes with an insufficient-row error on smaller CSVs
    max_allowed_lag = max(1, len(series) - max(horizon, 5) - 6)
    lags = [lag for lag in preferred_lags if lag <= max_allowed_lag]
    if not lags:
        lags = [1]
    rolling_window = min(preferred_window, max(2, max_allowed_lag))

    # Clamp horizon if series is very short
    horizon = min(horizon, max(3, len(series) // 3))

    stats = {
        "raw_valid_rows": raw_valid_rows,
        "missing_values": missing_values,
        "outliers_clipped": outliers_clipped,
    }
    return series, cadence, cadence_label, horizon, lags, rolling_window, stats


def make_features(series, lags, rolling_window, cadence):
    """Create lag, rolling-statistics, and calendar seasonality features for supervised learning."""
    features = pd.DataFrame(index=series.index)

    # Past values at specific lag offsets (lag_1, lag_2, ...)
    for lag in lags:
        features[f"lag_{lag}"] = series.shift(lag)

    # Shifted rolling mean & rolling std so we summarize recent history without leaking current target
    shifted = series.shift(1)
    features["rolling_mean"] = shifted.rolling(rolling_window).mean()
    features["rolling_std"] = shifted.rolling(rolling_window).std().fillna(0.0)

    # Calendar cycle features help capture recurring daily/weekly/seasonal patterns
    if cadence == "h":
        features["hour"] = series.index.hour
        features["day_of_week"] = series.index.day_of_week
    else:
        features["day_of_week"] = series.index.day_of_week
        features["month"] = series.index.month

    features["target"] = series
    return features.dropna()


def make_model():
    """Build a regularized Histogram-based Gradient Boosting Regressor resistant to sensor noise."""
    return HistGradientBoostingRegressor(
        random_state=42,
        max_depth=3,
        max_iter=80,
        learning_rate=0.05,
        min_samples_leaf=10,
        l2_regularization=3.0,
        early_stopping=False,
    )


def forecast_future(model, series, feature_columns, lags, rolling_window, cadence, horizon):
    """Predict multiple future steps recursively by feeding each step's prediction into the next."""
    history = series.copy()
    predictions = []
    timestamps = []
    step = pd.Timedelta(hours=1) if cadence == "h" else pd.Timedelta(days=1)

    for _ in range(horizon):
        next_timestamp = history.index[-1] + step
        row = {}

        # Populate lag features from the tail of `history`
        for lag in lags:
            row[f"lag_{lag}"] = float(history.iloc[-lag])

        # Populate rolling statistics from recent history
        recent_slice = history.iloc[-rolling_window:]
        row["rolling_mean"] = float(recent_slice.mean())
        row["rolling_std"] = float(recent_slice.std()) if len(recent_slice) > 1 else 0.0

        # Populate calendar features for `next_timestamp`
        if cadence == "h":
            row["hour"] = next_timestamp.hour
            row["day_of_week"] = next_timestamp.day_of_week
        else:
            row["day_of_week"] = next_timestamp.day_of_week
            row["month"] = next_timestamp.month

        # Predict and append to `history` for the next recursive step
        row_frame = pd.DataFrame([row], columns=feature_columns)
        prediction = float(model.predict(row_frame)[0])
        history.loc[next_timestamp] = prediction
        timestamps.append(next_timestamp)
        predictions.append(prediction)

    return pd.Series(predictions, index=timestamps, name=f"Forecast {series.name}")


def fetch_nearby_openaq(api_key, latitude, longitude):
    """Query OpenAQ v3 for the closest PM2.5 sensor within 25 km and return its last 7 days of hourly data."""
    headers = {"X-API-Key": api_key.strip()}

    # Search nearby monitoring locations within a 25 km radius for PM2.5 (parameters_id=2)
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
    location_response.raise_for_status()
    locations = location_response.json().get("results", [])

    # Collect candidate PM2.5 sensors and preserve provider & license attribution
    candidates = []
    for location in locations:
        for sensor in location.get("sensors", []):
            if sensor.get("parameter", {}).get("id") == 2:
                licenses = location.get("licenses") or []
                attribution = "See OpenAQ metadata"
                if licenses and isinstance(licenses[0], dict):
                    attribution = (licenses[0].get("attribution") or {}).get("name", attribution)
                candidates.append(
                    {
                        "distance": location.get("distance"),
                        "location": location.get("name") or "Unnamed location",
                        "provider": (location.get("provider") or {}).get("name", "Unknown provider"),
                        "attribution": attribution,
                        "sensor_id": sensor["id"],
                    }
                )

    if not candidates:
        raise ValueError("No OpenAQ PM2.5 sensor was found within 25 km of those coordinates.")

    # Select the physically nearest sensor
    closest = min(
        candidates,
        key=lambda item: item["distance"] if item["distance"] is not None else float("inf"),
    )

    # Fetch recent hourly measurements over the past 7 days
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

    live = pd.DataFrame(rows)
    if live.empty:
        raise ValueError(
            f"Sensor {sensor_id} ({closest['location']}) is registered nearby, "
            "but reported no hourly readings in the last 7 days."
        )
    live["timestamp"] = pd.to_datetime(live["timestamp"], errors="coerce", utc=True)
    live["value"] = pd.to_numeric(live["value"], errors="coerce")
    live = live.dropna().sort_values("timestamp")
    if live.empty:
        raise ValueError("The returned hourly records contained no usable timestamps and values.")

    return closest, live


# ==============================================================================
# SIDEBAR: DATA SOURCE (BUILT-IN DATASET + CUSTOM CSV UPLOAD) & CONFIGURATION
# ==============================================================================
with st.sidebar:
    st.header("1. Data Source")

    # Allow user to upload any custom CSV dataset whenever they want
    uploaded_file = st.file_uploader(
        "Upload custom CSV dataset (optional)",
        type=["csv"],
        help="Leave empty to use the predefined multi-city air quality dataset, or upload your own CSV.",
    )

    # Let the user switch cleanly between their uploaded file and the built-in dataset
    if uploaded_file is not None:
        source_choice = st.radio(
            "Active dataset",
            options=["Uploaded CSV", "Built-in Air Quality Dataset"],
            index=0,
        )
    else:
        source_choice = "Built-in Air Quality Dataset"
        st.caption("Using predefined multi-city dataset (`default_air_quality.csv`).")


# Load whichever dataset is active, with friendly error handling
if source_choice == "Uploaded CSV" and uploaded_file is not None:
    try:
        raw = pd.read_csv(uploaded_file)
        active_dataset_name = f"Uploaded CSV ({uploaded_file.name})"
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV file: {exc}")
        st.stop()
else:
    if not DEFAULT_DATASET_PATH.exists():
        st.error(
            "Built-in dataset `default_air_quality.csv` was not found next to `app.py`. "
            "Please upload a CSV file in the sidebar."
        )
        st.stop()
    raw = load_builtin_dataset(str(DEFAULT_DATASET_PATH))
    active_dataset_name = "Built-in Multi-City Air Quality Dataset (2023–2024)"

# Validate minimum column structure
if len(raw.columns) < 2:
    st.error("The dataset must contain at least two columns (a timestamp column and a numeric metric column).")
    st.stop()

# Automatically detect timestamp, numeric metrics, and optional city/location grouping columns
default_ts_col, detected_numeric_cols, detected_group_cols = detect_columns(raw)

with st.sidebar:
    st.header("2. Series Selection")

    # Timestamp column selector (auto-defaults to Date / timestamp)
    all_columns = raw.columns.tolist()
    ts_default_idx = all_columns.index(default_ts_col) if default_ts_col in all_columns else 0
    timestamp_column = st.selectbox("Timestamp column", all_columns, index=ts_default_idx)

    # Optional City / Location filter when dataset contains a grouping column like `City`
    selected_group_col = None
    selected_city = None
    filtered_frame = raw

    available_group_cols = [c for c in detected_group_cols if c != timestamp_column]
    if available_group_cols:
        selected_group_col = st.selectbox(
            "Filter by location / category",
            options=available_group_cols + ["None (Aggregate all rows)"],
            index=0,
        )
        if selected_group_col != "None (Aggregate all rows)":
            unique_groups = sorted(raw[selected_group_col].dropna().astype(str).unique().tolist())
            selected_city = st.selectbox(
                f"Select {selected_group_col}",
                options=["All (Daily Average)"] + unique_groups,
                index=0,
            )
            if selected_city != "All (Daily Average)":
                filtered_frame = raw[raw[selected_group_col].astype(str) == selected_city].copy()

    # Build safe list of numeric metric columns (never colliding with timestamp_column!)
    candidate_value_cols = [
        col for col in all_columns
        if col != timestamp_column and col != selected_group_col
        and pd.to_numeric(filtered_frame[col], errors="coerce").notna().sum() >= 5
    ]
    if not candidate_value_cols:
        st.error(
            "No numeric measurement columns with at least 5 valid numbers were found. "
            "Please upload a CSV containing numeric air quality or sensor values."
        )
        st.stop()

    # Default to PM2.5 if present, otherwise AQI, otherwise the first numeric column
    if "PM2.5" in candidate_value_cols:
        val_default_idx = candidate_value_cols.index("PM2.5")
    elif "AQI" in candidate_value_cols:
        val_default_idx = candidate_value_cols.index("AQI")
    else:
        val_default_idx = 0

    value_column = st.selectbox(
        "Target metric to forecast",
        options=candidate_value_cols,
        index=val_default_idx,
    )

    st.header("3. Forecast Settings")
    forecast_steps = st.slider(
        "Forecast horizon (steps ahead)",
        min_value=3,
        max_value=30,
        value=14,
        help="Number of future days (or hours, for hourly data) to predict.",
    )
    clip_outliers = st.checkbox(
        "Smooth extreme sensor spikes (IQR filter)",
        value=True,
        help="Clips extreme single-day sensor glitches (beyond 3.5x IQR) so they don't distort charts or training.",
    )


# ==============================================================================
# STEP 1: PREPARE & VALIDATE TIME SERIES
# ==============================================================================
unit = get_unit_for_column(value_column)
scope_label = (
    f"{selected_city}"
    if selected_city and selected_city != "All (Daily Average)"
    else "All Locations (Aggregated)"
)

try:
    series, cadence, cadence_label, horizon, lags, rolling_window, clean_stats = prepare_series(
        filtered_frame,
        timestamp_column,
        value_column,
        clip_outliers=clip_outliers,
        custom_horizon=forecast_steps,
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()


# ==============================================================================
# STEP 2: OVERVIEW KPIs, DATASET PREVIEW & MULTI-CITY SNAPSHOT
# ==============================================================================
st.markdown(f"**Active Source:** `{active_dataset_name}` &nbsp;|&nbsp; **Scope:** `{scope_label}` &nbsp;|&nbsp; **Metric:** `{value_column} ({unit})`")

kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
kpi_1.metric("Cleaned Observations", f"{len(series):,}")
kpi_2.metric("Detected Cadence", cadence_label.capitalize())
kpi_3.metric(f"Average {value_column}", f"{series.mean():.2f} {unit}")
kpi_4.metric("Forecast Horizon", f"{horizon} {cadence_label} steps")

with st.expander("Inspect raw dataset preview & data-cleaning summary"):
    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Valid Timestamp Rows", f"{clean_stats['raw_valid_rows']:,}")
    info_col2.metric("Missing Values Handled", f"{clean_stats['missing_values']:,}")
    info_col3.metric("Extreme Spikes Smoothed", f"{clean_stats['outliers_clipped']:,}")
    st.dataframe(filtered_frame.head(50))

# If dataset has multiple cities/locations, offer a quick cross-location comparison bar chart
if selected_group_col and selected_group_col in raw.columns:
    with st.expander(f"Compare average {value_column} across all {selected_group_col}s"):
        city_means = (
            raw.assign(_num=pd.to_numeric(raw[value_column], errors="coerce"))
            .groupby(selected_group_col)["_num"]
            .mean()
            .dropna()
            .sort_values(ascending=False)
            .rename(f"Mean {value_column} ({unit})")
        )
        st.bar_chart(city_means)


# ==============================================================================
# STEP 3: CLEANED HISTORY & RECURRING SEASONAL PATTERNS
# ==============================================================================
st.subheader(f"Historical {value_column} ({scope_label})")
st.line_chart(pd.DataFrame({f"Observed {value_column} ({unit})": series}))

st.subheader("Recurring Patterns")
pat_col1, pat_col2 = st.columns(2)

with pat_col1:
    if cadence == "h":
        # Average by hour of day (0..23 UTC) for hourly data
        hourly_pattern = series.groupby(series.index.hour).mean()
        hourly_pattern.index.name = "UTC Hour (0–23)"
        st.caption(f"Average {value_column} by Hour of Day")
        st.line_chart(pd.DataFrame({f"Mean {value_column}": hourly_pattern}))
    else:
        # Average by day of week (Monday=0 .. Sunday=6) for daily data
        weekday_names = ["0-Mon", "1-Tue", "2-Wed", "3-Thu", "4-Fri", "5-Sat", "6-Sun"]
        weekday_pattern = series.groupby(series.index.day_of_week).mean()
        weekday_pattern.index = [weekday_names[i] for i in weekday_pattern.index]
        weekday_pattern.index.name = "Day of Week"
        st.caption(f"Weekly Cycle — Mean {value_column} by Day of Week")
        st.bar_chart(pd.DataFrame({f"Mean {value_column}": weekday_pattern}))

with pat_col2:
    # Average by calendar month to reveal seasonal winter/summer trends
    month_names = [
        "01-Jan", "02-Feb", "03-Mar", "04-Apr", "05-May", "06-Jun",
        "07-Jul", "08-Aug", "09-Sep", "10-Oct", "11-Nov", "12-Dec",
    ]
    monthly_pattern = series.groupby(series.index.month).mean()
    monthly_pattern.index = [month_names[m - 1] for m in monthly_pattern.index]
    monthly_pattern.index.name = "Month"
    st.caption(f"Seasonal Cycle — Mean {value_column} by Month")
    st.bar_chart(pd.DataFrame({f"Mean {value_column}": monthly_pattern}))


# ==============================================================================
# STEP 4: MODEL EVIDENCE — RANDOM LEAKAGE CHECK VS. HONEST CHRONOLOGICAL HOLDOUT
# ==============================================================================
features = make_features(series, lags, rolling_window, cadence)
if len(features) < 8:
    st.warning("Not enough feature rows after lag warm-up to train the forecasting model. Upload more rows.")
    st.stop()

feature_columns = [col for col in features.columns if col != "target"]
X = features[feature_columns]
y = features["target"]

# 4A. Shuffled Random Split (demonstrates time-series data leakage)
X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(
    X, y, test_size=0.2, random_state=42
)
random_model = make_model()
random_model.fit(X_train_random, y_train_random)
random_predictions = random_model.predict(X_test_random)
random_mae = mean_absolute_error(y_test_random, random_predictions)
random_mix = bool(X_train_random.index.max() > X_test_random.index.min())

# 4B. Honest Chronological Split (trains strictly on past, evaluates strictly on latest 20%)
test_size = max(min(horizon, len(X) // 3), int(len(X) * 0.2))
test_size = max(2, min(test_size, len(X) - 4))
split_at = len(X) - test_size

X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]

chronological_model = make_model()
chronological_model.fit(X_train, y_train)
chronological_predictions = chronological_model.predict(X_test)
chronological_mae = mean_absolute_error(y_test, chronological_predictions)

# 4C. Naive Lag-1 Persistence Baseline (predicts next value == previous value)
persistence_predictions = X_test["lag_1"]
persistence_mae = mean_absolute_error(y_test, persistence_predictions)
chronology_ok = bool(X_train.index.max() < X_test.index.min())

st.subheader("Model Evidence & Validation")
val_left, val_right = st.columns(2)
with val_left:
    st.warning(f"Shuffled random split mixes future into training: **{random_mix}** (leaky evaluation)")
with val_right:
    st.success(f"Chronological holdout keeps training strictly before test: **{chronology_ok}** (honest evaluation)")

m_col1, m_col2, m_col3 = st.columns(3)
m_col1.metric(
    "Chronological Model MAE (Honest)",
    f"{chronological_mae:.2f} {unit}",
    delta=f"{chronological_mae - persistence_mae:+.2f} vs persistence",
    delta_color="inverse",
)
m_col2.metric("Lag-1 Persistence Baseline MAE", f"{persistence_mae:.2f} {unit}")
m_col3.metric("Random-Split MAE (Time-Leaked)", f"{random_mae:.2f} {unit}")

# Plot actual vs predicted on the chronological holdout window so model accuracy is visually transparent
holdout_comparison = pd.DataFrame(
    {
        f"Actual {value_column}": y_test,
        f"Model Predicted {value_column}": chronological_predictions,
    },
    index=y_test.index,
)
st.caption("Chronological Holdout Validation — Actual vs. Model Predictions on Unseen Recent Period")
st.line_chart(holdout_comparison)


# ==============================================================================
# STEP 5: FUTURE RECURSIVE FORECAST & USER-DEFINED WATCH LEVEL
# ==============================================================================
# Train final model on all available historical observations before forecasting ahead
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

st.subheader(f"Future {value_column} Forecast ({horizon} {cadence_label} steps)")
history_tail = max(horizon * 3, 30)
forecast_chart = pd.concat(
    [
        series.tail(history_tail).rename(f"Observed {value_column}"),
        forecast.rename(f"Forecast {value_column}"),
    ],
    axis=1,
)
st.line_chart(forecast_chart)

# Interactive user-defined watch threshold (defaults to the 90th percentile of historical values)
watch_default = float(round(series.quantile(0.90), 1))
watch_col1, watch_col2 = st.columns([1, 2])
with watch_col1:
    watch_level = st.number_input(
        f"User-defined watch level ({unit})",
        min_value=float(min(0.0, float(series.min()))),
        value=watch_default,
        step=1.0,
        help="Periods at or above this user-selected threshold are flagged below (not medical guidance).",
    )

forecast_table = pd.DataFrame(
    {
        f"forecast_{value_column.lower()}": forecast.round(2),
        "watch_level": round(watch_level, 2),
        "at_or_above_watch_level": forecast >= watch_level,
    }
)
forecast_table.index.name = "Forecast Timestamp (UTC)"

watch_count = int(forecast_table["at_or_above_watch_level"].sum())
with watch_col2:
    if watch_count:
        st.warning(
            f"**{watch_count}** of the next **{horizon}** forecast periods meet or exceed "
            f"your watch level of **{watch_level:.1f} {unit}**."
        )
    else:
        st.info(
            f"None of the next **{horizon}** forecast periods exceed "
            f"your watch level of **{watch_level:.1f} {unit}**."
        )

st.dataframe(forecast_table)

# Allow users to download the generated forecast table as a CSV
st.download_button(
    label="Download Forecast CSV",
    data=forecast_table.to_csv().encode("utf-8"),
    file_name=f"airsense_forecast_{value_column.lower()}.csv",
    mime="text/csv",
)


# ==============================================================================
# STEP 6: LIVE NEARBY SENSOR READINGS VIA OPENAQ API
# ==============================================================================
st.subheader("Live Nearby Sensor Readings (OpenAQ)")
st.caption(
    "Enter your OpenAQ v3 API key at runtime to pull the last 7 days of live hourly PM2.5 readings "
    "from the nearest monitoring station within 25 km. Your API key is never stored."
)

# Pre-fill coordinates if the selected city is in our known city dictionary; otherwise default to Los Angeles
default_lat, default_lon = CITY_COORDINATES.get(selected_city or "", (34.0522, -118.2437))

oaq_col1, oaq_col2, oaq_col3 = st.columns(3)
with oaq_col1:
    api_key = st.text_input("OpenAQ API key", type="password", placeholder="Paste OpenAQ v3 API key")
with oaq_col2:
    latitude_text = st.text_input("Latitude (WGS84)", value=f"{default_lat:.4f}")
with oaq_col3:
    longitude_text = st.text_input("Longitude (WGS84)", value=f"{default_lon:.4f}")

if st.button("Fetch nearby OpenAQ PM2.5"):
    if not api_key.strip():
        st.error("Please enter your OpenAQ API key first.")
    elif not latitude_text.strip() or not longitude_text.strip():
        st.error("Please enter both Latitude and Longitude coordinates.")
    else:
        try:
            latitude = float(latitude_text)
            longitude = float(longitude_text)
            if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
                raise ValueError("Latitude must be between -90 and 90, and Longitude between -180 and 180.")

            with st.spinner("Searching OpenAQ sensors within 25 km and fetching hourly readings..."):
                source, live = fetch_nearby_openaq(api_key, latitude, longitude)

            distance_km = (
                f" ({source['distance'] / 1000:.1f} km away)"
                if source.get("distance") is not None
                else ""
            )
            st.success(
                f"Loaded sensor **{source['sensor_id']}** at **{source['location']}**{distance_km}."
            )
            st.caption(
                f"Data source: OpenAQ | Provider: {source['provider']} | "
                f"Attribution: {source['attribution']}"
            )

            live_chart = live.set_index("timestamp")[["value"]].rename(
                columns={"value": "Recent Hourly PM2.5 (µg/m³)"}
            )
            st.line_chart(live_chart)
            st.dataframe(live.tail(24))
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "HTTP"
            if status == 401 or status == 403:
                st.error("OpenAQ authentication failed (HTTP 401/403). Please check that your OpenAQ v3 API key is valid.")
            else:
                st.error(f"OpenAQ server returned error ({status}): {exc}")
        except (ValueError, requests.exceptions.RequestException) as exc:
            st.error(f"OpenAQ lookup notice: {exc}")
