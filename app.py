import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Air Sense Nepal - Pokhara PM2.5 Dashboard",
    layout="wide",
    page_icon=":cloud:",
)
st.title("Air Sense Nepal - Pokhara PM2.5 Forecasting Dashboard")
st.caption(
    "This app works with PM2.5 concentration measurements, not a precomputed AQI. "
    "Its watch level is user-defined and is not medical guidance."
)


def prepare_series(frame, timestamp_column, value_column):
    selected = frame[[timestamp_column, value_column]].copy()
    selected.columns = ["timestamp", "pm25"]
    selected["timestamp"] = pd.to_datetime(
        selected["timestamp"], errors="coerce", utc=True
    )
    selected["pm25"] = pd.to_numeric(selected["pm25"], errors="coerce")
    selected = selected.dropna()

    series = selected.groupby("timestamp")["pm25"].mean().sort_index()
    if len(series) < 3:
        raise ValueError("The selected columns contain fewer than three usable rows.")

    median_gap = series.index.to_series().diff().dropna().median()
    if median_gap <= pd.Timedelta(hours=6):
        cadence = "h"
        cadence_label = "hourly"
        horizon = 24
        lags = [1, 2, 3, 24, 168]
        rolling_window = 24
    else:
        cadence = "D"
        cadence_label = "daily"
        horizon = 7
        lags = [1, 2, 3, 7, 14]
        rolling_window = 7

    series = series.resample(cadence).mean().interpolate().dropna()
    series.name = "pm25"
    return series, cadence, cadence_label, horizon, lags, rolling_window


def make_features(series, lags, rolling_window):
    features = pd.DataFrame(index=series.index)
    for lag in lags:
        features[f"lag_{lag}"] = series.shift(lag)
    features["rolling_mean"] = series.shift(1).rolling(rolling_window).mean()
    features["target"] = series
    return features.dropna()


def make_model():
    return HistGradientBoostingRegressor(
        random_state=42,
        min_samples_leaf=5,
        early_stopping=False,
    )


def forecast_future(model, series, feature_columns, lags, rolling_window, cadence, horizon):
    history = series.copy()
    predictions = []
    timestamps = []
    step = pd.Timedelta(hours=1) if cadence == "h" else pd.Timedelta(days=1)

    for _ in range(horizon):
        next_timestamp = history.index[-1] + step
        row = {}
        for lag in lags:
            row[f"lag_{lag}"] = history.iloc[-lag]
        row["rolling_mean"] = history.iloc[-rolling_window:].mean()
        row_frame = pd.DataFrame([row], columns=feature_columns)
        prediction = float(model.predict(row_frame)[0])
        history.loc[next_timestamp] = prediction
        timestamps.append(next_timestamp)
        predictions.append(prediction)

    return pd.Series(predictions, index=timestamps, name="Forecast PM2.5")


def fetch_nearby_openaq(api_key, latitude, longitude):
    headers = {"X-API-Key": api_key}
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

    closest = min(
        candidates,
        key=lambda item: item["distance"]
        if item["distance"] is not None
        else float("inf"),
    )

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
        raise ValueError("The selected sensor returned no hourly rows for the last week.")
    live["timestamp"] = pd.to_datetime(live["timestamp"], errors="coerce", utc=True)
    live["value"] = pd.to_numeric(live["value"], errors="coerce")
    live = live.dropna().sort_values("timestamp")
    if live.empty:
        raise ValueError("The returned hourly rows had no usable timestamps and values.")

    return closest, live


uploaded_file = st.file_uploader("Upload the Pokhara CSV", type=["csv"])
if uploaded_file is None:
    st.info("Upload the Pokhara CSV to begin.")
    st.stop()

try:
    raw = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"The CSV could not be read: {exc}")
    st.stop()

if len(raw.columns) < 2:
    st.error("The CSV needs at least two columns.")
    st.stop()

timestamp_column = st.selectbox("Timestamp column", raw.columns.tolist())
value_column = st.selectbox("PM2.5 concentration column", raw.columns.tolist())
if timestamp_column == value_column:
    st.warning("Choose different timestamp and PM2.5 columns.")
    st.stop()

try:
    series, cadence, cadence_label, horizon, lags, rolling_window = prepare_series(
        raw, timestamp_column, value_column
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

minimum_rows = max(lags) + horizon + 20
if len(series) < minimum_rows:
    st.error(
        f"The cleaned {cadence_label} series needs at least {minimum_rows} rows "
        f"for these lags and forecast horizon. It currently has {len(series)}."
    )
    st.stop()

summary_1, summary_2, summary_3 = st.columns(3)
summary_1.metric("Usable observations", len(series))
summary_2.metric("Detected cadence", cadence_label)
summary_3.metric("Forecast horizon", f"{horizon} {cadence_label} steps")

st.subheader("Historical PM2.5")
st.line_chart(pd.DataFrame({"Observed PM2.5": series}))

st.subheader("Recurring pattern")
if cadence == "h":
    pattern = series.groupby(series.index.hour).mean()
    pattern.index.name = "UTC hour"
else:
    pattern = series.groupby(series.index.day_of_week).mean()
    pattern.index.name = "Weekday number, Monday=0"
st.line_chart(pd.DataFrame({"Mean PM2.5": pattern}))

features = make_features(series, lags, rolling_window)
feature_columns = [column for column in features.columns if column != "target"]
X = features[feature_columns]
y = features["target"]

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
persistence_predictions = X_test["lag_1"]
persistence_mae = mean_absolute_error(y_test, persistence_predictions)
chronology_ok = X_train.index.max() < X_test.index.min()

st.subheader("Honest future holdout")
st.success(f"Chronological split keeps train before test: {chronology_ok}")
metric_1, metric_2, metric_3 = st.columns(3)
metric_1.metric("Random MAE", f"{random_mae:.2f} µg/m³")
metric_2.metric("Chronological MAE", f"{chronological_mae:.2f} µg/m³")
metric_3.metric("Persistence MAE", f"{persistence_mae:.2f} µg/m³")

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

st.subheader("Future PM2.5 forecast")
forecast_chart = pd.concat(
    [series.tail(horizon * 3).rename("Observed PM2.5"), forecast], axis=1
)
st.line_chart(forecast_chart)

watch_default = float(round(series.quantile(0.90), 1))
watch_level = st.number_input(
    "User-defined watch level (µg/m³)",
    min_value=0.0,
    value=watch_default,
    step=1.0,
)
forecast_table = pd.DataFrame({"forecast_pm25": forecast})
forecast_table["at_or_above_watch_level"] = forecast_table["forecast_pm25"] >= watch_level
st.dataframe(forecast_table.round(2))
watch_count = int(forecast_table["at_or_above_watch_level"].sum())
if watch_count:
    st.warning(f"{watch_count} forecast periods meet or exceed the watch level.")
else:
    st.info("No forecast periods meet the current watch level.")

st.subheader("Recent OpenAQ PM2.5 context")
st.caption("Enter the API key at runtime. It is not stored by this app.")
api_key = st.text_input("OpenAQ API key", type="password")
latitude_text = st.text_input("Latitude from the Pokhara data or metadata")
longitude_text = st.text_input("Longitude from the Pokhara data or metadata")

if st.button("Fetch nearby OpenAQ PM2.5"):
    if not api_key or not latitude_text or not longitude_text:
        st.error("Enter an API key, latitude, and longitude.")
    else:
        try:
            latitude = float(latitude_text)
            longitude = float(longitude_text)
            with st.spinner("Fetching recent OpenAQ measurements"):
                source, live = fetch_nearby_openaq(api_key, latitude, longitude)
            st.success(
                f"Loaded sensor {source['sensor_id']} at {source['location']}."
            )
            st.caption(
                f"Data source: OpenAQ. Provider: {source['provider']}. "
                f"Attribution: {source['attribution']}."
            )
            live_chart = live.set_index("timestamp")[["value"]]
            live_chart.columns = ["Recent hourly PM2.5"]
            st.line_chart(live_chart)
            st.dataframe(live.tail(24))
        except (ValueError, requests.exceptions.RequestException) as exc:
            st.error(f"OpenAQ request failed: {exc}")
