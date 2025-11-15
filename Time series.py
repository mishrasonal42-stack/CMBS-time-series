import plotly.graph_objects as go
import pandas as pd
import numpy as np



st.title("CMBS Property Time Series + 5-Year Scenario Forecast")

# ----------------------------------------------------------
# 1. FILE UPLOAD
# ----------------------------------------------------------
file = st.file_uploader("Upload CMBS_property_data.xlsx", type=["xlsx"])

if file:
    df = pd.read_excel(file)

    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())

    # Select property
    property_name = st.selectbox("Select Property", df["Property Name"].unique())
    pdf = df[df["Property Name"] == property_name].sort_values("Year")

    st.write(f"### Historical Data for: {property_name}")
    st.dataframe(pdf)

    # ----------------------------------------------------------
    # 2. PURE NUMPY LINEAR REGRESSION
    # ----------------------------------------------------------
    def simple_linear_regression(y):
        """
        Returns slope & intercept for y = a*x + b using least squares.
        x is assumed as [0, 1, 2, ...].
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
    future_years = np.arange(pdf["Year"].max()+1, pdf["Year"].max()+1+years_forward)

    # Forecasts using trend
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
        "Value_Down": val_fc * 0.90
    })

    st.write("### 5-Year Scenario Forecast")
    st.dataframe(scenarios)

    # ----------------------------------------------------------
    # 4. PLOTS USING PLOTLY (fully Streamlit-compatible)
    # ----------------------------------------------------------

    # NOI
    fig_noi = go.Figure()
    fig_noi.add_trace(go.Scatter(x=pdf["Year"], y=pdf["NOI"], mode="lines+markers", name="Historical"))
    fig_noi.add_trace(go.Scatter(x=future_years, y=scenarios["NOI_Base"], name="Base Case"))
    fig_noi.add_trace(go.Scatter(x=future_years, y=scenarios["NOI_Up"], name="Upside"))
    fig_noi.add_trace(go.Scatter(x=future_years, y=scenarios["NOI_Down"], name="Downside"))
    fig_noi.update_layout(title="NOI Projection", xaxis_title="Year", yaxis_title="NOI")
    st.plotly_chart(fig_noi)

    # Occupancy
    fig_occ = go.Figure()
    fig_occ.add_trace(go.Scatter(x=pdf["Year"], y=pdf["Occupancy"], mode="lines+markers", name="Historical"))
    fig_occ.add_trace(go.Scatter(x=future_years, y=scenarios["Occ_Base"], name="Base Case"))
    fig_occ.add_trace(go.Scatter(x=future_years, y=scenarios["Occ_Up"], name="Upside"))
    fig_occ.add_trace(go.Scatter(x=future_years, y=scenarios["Occ_Down"], name="Downside"))
    fig_occ.update_layout(title="Occupancy Projection", xaxis_title="Year", yaxis_title="Occupancy %")
    st.plotly_chart(fig_occ)

    # Value
    fig_val = go.Figure()
    fig_val.add_trace(go.Scatter(x=pdf["Year"], y=pdf["Value"], mode="lines+markers", name="Historical"))
    fig_val.add_trace(go.Scatter(x=future_years, y=scenarios["Value_Base"], name="Base Case"))
    fig_val.add_trace(go.Scatter(x=future_years, y=scenarios["Value_Up"], name="Upside"))
    fig_val.add_trace(go.Scatter(x=future_years, y=scenarios["Value_Down"], name="Downside"))
    fig_val.update_layout(title="Property Value Projection", xaxis_title="Year", yaxis_title="Value")
    st.plotly_chart(fig_val)
