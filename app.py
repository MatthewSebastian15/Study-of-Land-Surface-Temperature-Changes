import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import predict_temperature

DATA_PATH = "data_average_surface_temperature.csv"

st.title("🌍 Surface Temperature Prediction")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Average surface temperature year'])
    df['date'] = pd.to_datetime(df['year'].astype(str) + '-01-01')
    return df

df = load_data()

st.sidebar.header("🔧 Settings")
countries = sorted(df['Entity'].unique())
selected_country = st.sidebar.selectbox("Select a Country", countries)
show_predicted = st.sidebar.checkbox("Show Predicted 2000 – 2024", value=True)
show_forecast = st.sidebar.button("Show Forecast 2025 – 2026")

country_data = df[df['Entity'] == selected_country].copy()
historical = country_data[country_data['year'] < 2025]
forecast_df = predict_temperature(df, 2025, 2026)
forecast_df = forecast_df[forecast_df['Entity'] == selected_country]
forecast_df['date'] = pd.to_datetime(forecast_df['year'].astype(str) + '-01-01')

st.subheader(f"📊 Actual Temperature for {selected_country} (2000–2024)")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(historical['date'], historical['Average surface temperature year'], 'bo-', label='Actual Temperature')

if show_predicted and 'Temperature anomaly' in historical.columns:
    predicted = historical['Average surface temperature year'] + historical['Temperature anomaly']
    ax.plot(historical['date'], predicted, 'rx--', label='Predicted Temperature')

if show_forecast:
    ax.plot(forecast_df['date'], forecast_df['Forecast'], 'gD-', label='Forecasted Temperature')

ax.set_xlabel("Year")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Surface Temperature")
ax.legend()
ax.grid(True)
st.pyplot(fig)
