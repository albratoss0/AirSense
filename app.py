import pandas as pd
import streamlit as st
st.set_page_config(page_title= "Air Sense Nepal - AI Powered Air Quality Intelligence", layout="wide", page_icon=":cloud:")
st.title("Air Sense Nepal - AI Powered Air Quality Intelligence")
st.caption("This is a web application that provide real-time air quality information and predictions for different locations in Nepal using AI models.")
uploaded_file = st.file_uploader("Upload an air quality CSV file", type=["csv"])
if uploaded_file is None:
    st.info ("Please upload a csv file to proceed.")
    st.stop()
try:
    raw = pd.read_csv(uploaded_file)
except Exception as exv:
    st.error(f"Could not read the uploaded CSV file: {exv}")
    st.stop()
    if len (raw.columns) < 2:
        st.error("The uploaded CSV file must have at least two columns.")
        st.stop()
        timestamp_column = st.selectbox("Timestamp column", raw.columns.tolist())
        value_column = st.selectbox("Air quality value column", raw.columns.tolist())
        if timestamp_column == value_column: 
            st.warning("Please select different columns for timestamp and air quality value.")
            st.stop()
            def prepare_series(frame, timestamp_column, value_column):
                selected = frame[[timestamp_column, value_column]].copy()
                selected.columns = ["timestamp", "value"]
                selected["timestamp"] = pd.to_datetime(selected["timestamp"], errors= "coerce", utc=True)
                selected["value"] = pd.to_numeric(selected["value"], errors= "coerce")
                selected = selected.dropna()
                series = selected.groupby ("timestamp")["value"].mean().sort_index()
                if len(series) < 3:
                    raise ValueError("The Selected columns contain fewer than three usable rows")
                median_gap =series.index.to_series().diff().dropna().median()
                if median_gap <= pd.Timedelta(hours=6):
                    cadence = "H"
                    candence_label = "hourly"
                    horizon = 27
                    lags = [1, 2, 3, 24, 168]
                    rolling_window = 24
                else:
                    candence = "D"
                    candence_label = "daily"
                    horizon = 7
                    lags = [1, 2, 3, 7, 14]
                    rolling_window = 7

                    seires = series.resample(candence).mean().interpolate().dropna()
                    series.name = "value"
                    return series, cadence, candence_label, horizon, lags, rolling_window
                try: 
                    series, cadence, cadence_label, horizon, lags, rolling_window = prepare_series(raw, timestamp_column, value_column)
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()
                    minimum_rows = max(lags) + horizon + 20
                    if len(series) < minimum_rows:
                        st.error(f"The cleaned {cadence_label} series has {len(series)} rows, but at least {minimum_rows}series needs at least {minimum_rows} rows for the selected lags and horizon." f"for these lags and forecast horizon. It is currnetly has {len(series)}.")
                        st.stop()
                        summary_1, summary_2, summary_3 = st.columns(3)
                        summary_1.metrics ("Usable Observations", len(series))
                        summary_2.metric("Cadence", cadence_label)
                        summary_3.metric("Forecast Horizon ", f"{horizon}{candence_label} steps")