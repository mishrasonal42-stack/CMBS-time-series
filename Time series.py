import streamlit as st
import pandas as pd
import numpy as np

st.title("CMBS Property Time Series + 5-Year Scenario Forecast (CSV Only Version)")

# ----------------------------------------------------------
# 1. FILE UPLOAD (CSV ONLY)
# ----------------------------------------------------------
file = st.file_uploader("Upload CMBS_property_data.csv", type=["csv"])

if not file:
    st.warning("Please upload a CSV file (Excel .xlsx is not supported).")
    st.stop()

# Load CSV safely
try:
    df = pd.read_csv(file)
except Exception as e:
    st.error(f"Failed to read CSV file: {e}")
    st.stop()

st.write("### Preview of Uploaded Data")
st.dataframe(df.head())

# ----------------------------------------------------------
# SELECT PROPERTY
# ----------------------------------------------------------
property_name = st.selectbox("Select Property", df["Property Name"].unique())
pdf = df[df["Property Name"] == property_name].sort_values("Year")

st.write(f"### Historical Data for: {property_name}")
st.dataframe(pdf)

# ----------------------------------------------------------
# TREND MODEL USING PURE NUMPY
# ----------------------------------------------------------
def simple_linear_regression(y):
    x = np.arange(len(y))
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope, intercept

def forecast_trend(series, years_forward=5):
    slope, intercept = simple_linear_regression(series.values)
    future_x = np.arange(len(series), len(series) + years_forward)
    return intercept + slope * future_x

# Series
ts_noi = pdf["NOI"]
ts_occ = pdf["Occupancy"]
ts_val = pdf["Value"]

years_forward = 5
future_years = np.arange(pdf["Year"].max() + 1,
                         pdf["Year"].max() + 1 + years_forward)

# Forecasts
noi_fc = forecast_trend(ts_noi, years_forward)
occ_fc = forecast_trend(ts_occ, years_forward)
val_fc = forecast_trend(ts_val, years_forward)

# ----------------------------------------------------------
# SCENARIO ANALYSIS
# ----------------------------------------------------------
scenarios = pd.DataFrame({
    "Year": future_years,

    "NOI_Base": noi_fc,
    "NOI_Up": noi_fc * 1.05,
    "NOI_Down": noi_fc * 0.95,

    "Occ_Base": occ_fc,
    "Occ_Up": np.minimum(100, occ_fc * 1.02),
    "Occ_Down": occ_fc * 0.97,

    "Value_Base": val_fc,
    "Value_Up": val_fc * 1.08,
    "Value_Down": val_fc * 0.90
})

st.write("### 5-Year Scenario Forecast")
st.dataframe(scenarios)

# ----------------------------------------------------------
# STREAMLIT NATIVE CHARTS
# ----------------------------------------------------------

# NOI Chart
st.subheader("NOI Projection")
noi_chart = pd.DataFrame({
    "Year": list(pdf["Year"]) + list(future_years),
    "Historical NOI": list(ts_noi.values) + [None]*len(future_years),
    "Base Case": [None]*len(ts_noi) + list(scenarios["NOI_Base"]),
    "Upside": [None]*len(ts_noi) + list(scenarios["NOI_Up"]),
    "Downside": [None]*len(ts_noi) + list(scenarios["NOI_Down"])
}).set_index("Year")
st.line_chart(noi_chart)

# Occupancy Chart
st.subheader("Occupancy Projection")
occ_chart = pd.DataFrame({
    "Year": list(pdf["Year"]) + list(future_years),
    "Historical Occupancy": list(ts_occ.values) + [None]*len(future_years),
    "Base Case": [None]*len(ts_occ) + list(scenarios["Occ_Base"]),
    "Upside": [None]*len(ts_occ) + list(scenarios["Occ_Up"]),
    "Downside": [None]*len(ts_occ) + list(scenarios["Occ_Down"])
}).set_index("Year")
st.line_chart(occ_chart)

# Value Chart
st.subheader("Value Projection")
val_chart = pd.DataFrame({
    "Year": list(pdf["Year"]) + list(future_years),
    "Historical Value": list(ts_val.values) + [None]*len(future_years),
    "Base Case": [None]*len(ts_val) + list(scenarios["Value_Base"]),
    "Upside": [None]*len(ts_val) + list(scenarios["Value_Up"]),
    "Downside": [None]*len(ts_val) + list(scenarios["Value_Down"])
}).set_index("Year")
st.line_chart(val_chart)
