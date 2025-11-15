import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------
# 1. LOAD CMBS PROPERTY-LEVEL DATA
# -----------------------------------------------------
df = pd.read_excel("CMBS_property_data.xlsx")

# Expected columns (modify if needed):
# 'Property Name', 'Year', 'NOI', 'Occupancy', 'Value', 'DSCR'
# If your sheet has other names, rename here:
# df = df.rename(columns={'Net Operating Income':'NOI', 'Appraised Value':'Value'})


# -----------------------------------------------------
# 2. SELECT A PROPERTY FOR ANALYSIS
# -----------------------------------------------------
property_name = df['Property Name'].unique()[0]   # or manually set a name
pdf = df[df['Property Name'] == property_name].sort_values('Year')

print(f"\nAnalyzing Property: {property_name}\n")
print(pdf)


# -----------------------------------------------------
# 3. TIME SERIES MODEL – TREND FORECAST (LINEAR REGRESSION)
# -----------------------------------------------------
def forecast_linear(series, years_forward=5):
    """Linear regression time-series projection."""
    X = np.array(series.index).reshape(-1, 1)
    y = np.array(series.values)
    
    model = LinearRegression().fit(X, y)
    future_index = np.arange(len(series), len(series) + years_forward).reshape(-1, 1)
    forecast = model.predict(future_index)

    return forecast


# Build time series indexed from 0..n
ts_noi = pd.Series(pdf['NOI'].values)
ts_occ = pd.Series(pdf['Occupancy'].values)
ts_value = pd.Series(pdf['Value'].values)

years_forward = 5

noi_forecast = forecast_linear(ts_noi, years_forward)
occ_forecast = forecast_linear(ts_occ, years_forward)
value_forecast = forecast_linear(ts_value, years_forward)

future_years = np.arange(pdf['Year'].max()+1, pdf['Year'].max()+1+years_forward)


# -----------------------------------------------------
# 4. SCENARIO ANALYSIS
# -----------------------------------------------------
scenarios = pd.DataFrame({
    "Year": future_years,
    
    # base-case = linear model
    "NOI_Base": noi_forecast,
    "Occ_Base": occ_forecast,
    "Value_Base": value_forecast,
    
    # downside assumptions
    "NOI_Down": noi_forecast * 0.95,     # -5% NOI shock
    "Occ_Down": occ_forecast * 0.97,     # -3% occupancy
    "Value_Down": value_forecast * 0.90, # -10% valuation
    
    # upside assumptions
    "NOI_Up": noi_forecast * 1.05,        # +5% NOI lift
    "Occ_Up": np.minimum(100, occ_forecast * 1.02), # +2% occupancy capped at 100
    "Value_Up": value_forecast * 1.08     # +8% valuation
})

print("\n----- 5-Year Scenario Forecast -----\n")
print(scenarios)


# -----------------------------------------------------
# 5. PLOTS (NOI, Occupancy, Value)
# -----------------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(pdf['Year'], pdf['NOI'], label='Historical')
plt.plot(scenarios['Year'], scenarios['NOI_Base'], label='Base Case')
plt.plot(scenarios['Year'], scenarios['NOI_Down'], label='Downside')
plt.plot(scenarios['Year'], scenarios['NOI_Up'], label='Upside')
plt.title(f"NOI Projection – {property_name}")
plt.xlabel("Year")
plt.ylabel("NOI")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10,5))
plt.plot(pdf['Year'], pdf['Occupancy'], label='Historical')
plt.plot(scenarios['Year'], scenarios['Occ_Base'], label='Base Case')
plt.plot(scenarios['Year'], scenarios['Occ_Down'], label='Downside')
plt.plot(scenarios['Year'], scenarios['Occ_Up'], label='Upside')
plt.title(f"Occupancy Projection – {property_name}")
plt.xlabel("Year")
plt.ylabel("Occupancy %")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10,5))
plt.plot(pdf['Year'], pdf['Value'], label='Historical')
plt.plot(scenarios['Year'], scenarios['Value_Base'], label='Base Case')
plt.plot(scenarios['Year'], scenarios['Value_Down'], label='Downside')
plt.plot(scenarios['Year'], scenarios['Value_Up'], label='Upside')
plt.title(f"Property Value Projection – {property_name}")
plt.xlabel("Year")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
