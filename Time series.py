import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO, BytesIO

st.title("CMBS Property Time Series + 5-Year Scenario Forecast (Streamlit Native)")

# ----------------------------------------------------------
# 1. FILE UPLOAD
# ----------------------------------------------------------
file = st.file_uploader("Upload CMBS_property_data.xlsx", type=["xlsx", "csv"])

if file:
    # If CSV → load normally
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)

    # If XLSX → try openpyxl, else convert to CSV using fallback
    elif file.name.endswith(".xlsx"):
        try:
            df = pd.read_excel(file, engine="openpyxl")
        except Exception:
            st.warning("openpyxl not available — converting Excel to CSV")

            from xlsx2csv import Xlsx2csv

            # Convert XLSX → CSV in memory
            csv_buffer = StringIO()
            Xlsx2csv(file).convert(csv_buffer)
            csv_buffer.seek(0)
            df = pd.read_csv(csv_buffer)

    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())
