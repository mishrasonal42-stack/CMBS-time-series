import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.title("CMBS Property Time Series + 5-Year Scenario Forecast")

# ----------------------------------------------------------
# 1. FILE UPLOAD
# ----------------------------------------------------------
file = st.file_uploader("Upload CMBS_property_data.xlsx", type=["xlsx"])

if file is not None:

    # Read Excel
    df = pd.read_excel(file)

    # Validate required columns
    required_cols = ["Property Name", "Year", "NOI", "Occupancy", "Value"]
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # Property selection
    properties = df["Property Name"].unique()
    property_name = st.selectbox("Select Property", properties)

    pdf = df[df["Property Name"] == property_name].sort_values("Year")

    st.subheader(f"Historical Data for: {property_name}")
    st.dataframe(pdf)

    # ----------------------------------------------------------
    # 2. PURE NUMPY LINEAR REGRESSION (NO sklearn needed)
    # ----------------------------------------------------------
    def simple_linear_regression(y):
        """
        Computes slope & intercept for y = a*x + b using least squares.
        """
        x = np.arange(len(y))
        A = np.vstack([x, np.ones(len(x))]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        return slope, intercept

    def forecast_trend(series, years_forward=5):
        slope, intercept = simple_linear_regression(series.values)
        future_x = np.arange(len(series), len(series) + years_forward)
        return intercept + slope * future_x

    # Historical series
    ts_noi = pdf["NOI"]
    ts_occ = pdf["Occupancy"]
    ts_val = pdf["Value"]

    years_forward = 5
    future_years = np.arange(pdf["Year"].max() + 1, pdf["Year"].max() + 1 + years_forward)

    # Forecasts using NumPy linear regression
    noi_fc = forecast_trend(ts_noi, years_forward)
    occ_fc = forecast_trend(ts_occ, years_forward)
    val_fc = forecast_trend(ts_val, years_forward)

    # ----------------------------------------------------------
    # 3. SCENARIO ANALYSIS
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
        "Value_Down": val_fc * 0.90,
    })

    st.subheader("5-Year Scenario Forecast")
    st.dataframe(scenarios)

    # ----------------------------------------------------------
    # 4. PLOTLY CHARTS (Streamlit compatible)
    # ----------------------------------------------------------

    # NOI Projection
    fig_noi = go.Figure()
    fig_noi.add_trace(go.Scatter(x=pdf["Year"], y=pdf["NOI"], mode="lines+markers", name="Historical"))
    fig_noi.add_trace(go.Scatter(x=future_years, y=scenarios["NOI_Base"], name="Base Case"))
    fig_noi.add_trace(go.Scatter(x=future_years, y=scenarios["NOI_Up"], name="Upside"))
    fig_noi.add_trace(go.Scatter(x=future_years, y=scenarios["NOI_Down"], name="Downside"))
    fig_noi.update_layout(title="NOI Projection", xaxis_title="Year", yaxis_title="NOI")
    st.plotly_chart(fig_noi)

    # Occupancy Projection
    fig_occ = go.Figure()
    fig_occ.add_trace(go.Scatter(x=pdf["Year"], y=pdf["Occupancy"], mode="lines+markers", name="Historical"))
    fig_occ.add_trace(go.Scatter(x=future_years, y=scenarios["Occ_Base"], name="Base Case"))
    fig_occ.add_trace(go.Scatter(x=future_years, y=scenarios["Occ_Up"], name="Upside"))
    fig_occ.add_trace(go.Scatter(x=future_years, y=scenarios["Occ_Down"], name="Downside"))
    fig_occ.update_layout(title="Occupancy Projection", xaxis_title="Year", yaxis_title="Occupancy %")
    st.plotly_chart(fig_occ)

    # Value Projection
    fig_val = go.Figure()
    fig_val.add_trace(go.Scatter(x=pdf["Year"], y=pdf["Value"], mode="lines+markers", name="Historical"))
    fig_val.add_trace(go.Scatter(x=future_years, y=scenarios["Value_Base"], name="Base Case"))
    fig_val.add_trace(go.Scatter(x=future_years, y=scenarios["Value_Up"], name="Upside"))
    fig_val.add_trace(go.Scatter(x=future_years, y=scenarios["Value_Down"], name="Downside"))
    fig_val.update_layout(title="Property Value Projection", xaxis_title="Year", yaxis_title="Value")
    st.plotly_chart(fig_val)
