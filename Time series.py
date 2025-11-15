import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

st.title("CMBS Property Time Series + Scenario Forecast")

# ----------------------------------------------------------
# 1. UPLOAD EXCEL
# ----------------------------------------------------------
file = st.file_uploader("Upload CMBS_property_data.xlsx", type=["xlsx"])

if file:
    df = pd.read_excel(file)

    # ----------------------------------------------------------
    # 2. PREPARE COLUMNS (rename if needed)
    # ----------------------------------------------------------
    # Expected columns:
    # Property Name | Year | NOI | Occupancy | Value

    st.write("### Raw Data Preview")
    st.dataframe(df.head())

    # Select property
    property_name = st.selectbox("Select Property", df["Property Name"].unique())
    pdf = df[df["Property Name"] == property_name].sort_values("Year")

    st.write(f"### Historical Data: {property_name}")
    st.dataframe(pdf)

    # ----------------------------------------------------------
    # 3. FORECAST FUNCTION
    # ----------------------------------------------------------
    def forecast_linear(series, years_forward=5):
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values
        model = LinearRegression().fit(X, y)
        future_index = np.arange(len(series), len(series) + years_forward).reshape(-1, 1)
        return model.predict(future_index)

    years_forward = 5
    future_years = np.arange(pdf['Year'].max() + 1, pdf['Year'].max() + 1 + years_forward)

    # Build series
    ts_noi = pd.Series(pdf["NOI"].values)
    ts_occ = pd.Series(pdf["Occupancy"].values)
    ts_val = pd.Series(pdf["Value"].values)

    noi_fc = forecast_linear(ts_noi, years_forward)
    occ_fc = forecast_linear(ts_occ, years_forward)
    val_fc = forecast_linear(ts_val, years_forward)

    # ----------------------------------------------------------
    # 4. SCENARIOS
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

    st.write("### 5-Year Scenario Forecast")
    st.dataframe(scenarios)

    # ----------------------------------------------------------
    # 5. PLOTLY CHARTS
    # ----------------------------------------------------------

    # ---- NOI Chart ----
    fig_noi = go.Figure()
    fig_noi.add_trace(go.Scatter(x=pdf["Year"], y=pdf["NOI"], mode="lines+markers", name="Historical"))
    fig_noi.add_trace(go.Scatter(x=future_years, y=scenarios["NOI_Base"], name="Base Case"))
    fig_noi.add_trace(go.Scatter(x=future_years, y=scenarios["NOI_Up"], name="Upside"))
    fig_noi.add_trace(go.Scatter(x=future_years, y=scenarios["NOI_Down"], name="Downside"))
    fig_noi.update_layout(title="NOI Projection", xaxis_title="Year", yaxis_title="NOI")
    st.plotly_chart(fig_noi)

    # ---- Occupancy Chart ----
    fig_occ = go.Figure()
    fig_occ.add_trace(go.Scatter(x=pdf["Year"], y=pdf["Occupancy"], mode="lines+markers", name="Historical"))
    fig_occ.add_trace(go.Scatter(x=future_years, y=scenarios["Occ_Base"], name="Base Case"))
    fig_occ.add_trace(go.Scatter(x=future_years, y=scenarios["Occ_Up"], name="Upside"))
    fig_occ.add_trace(go.Scatter(x=future_years, y=scenarios["Occ_Down"], name="Downside"))
    fig_occ.update_layout(title="Occupancy Projection", xaxis_title="Year", yaxis_title="Occupancy %")
    st.plotly_chart(fig_occ)

    # ---- Value Chart ----
    fig_val = go.Figure()
    fig_val.add_trace(go.Scatter(x=pdf["Year"], y=pdf["Value"], mode="lines+markers", name="Historical"))
    fig_val.add_trace(go.Scatter(x=future_years, y=scenarios["Value_Base"], name="Base Case"))
    fig_val.add_trace(go.Scatter(x=future_years, y=scenarios["Value_Up"], name="Upside"))
    fig_val.add_trace(go.Scatter(x=future_years, y=scenarios["Value_Down"], name="Downside"))
    fig_val.update_layout(title="Property Value Projection", xaxis_title="Year", yaxis_title="Value")
    st.plotly_chart(fig_val)
