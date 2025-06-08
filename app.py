import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import predict_temperature

# Path ke file CSV
DATA_PATH = "data_average_surface_temperature.csv"

# Judul
st.title("Surface Temperature Prediction: Historical and Future")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df['date'] = pd.to_datetime(df['year'].astype(str) + '-01-01')
    return df

df = load_data()

# Pilihan negara
countries = sorted(df['Entity'].unique())
selected_country = st.selectbox("Pilih Negara", countries)

# Filter berdasarkan negara
country_data = df[df['Entity'] == selected_country]
historical = country_data[country_data['year'] < 2025]

# Prediksi forecast
forecast_df = predict_temperature(country_data, 2025, 2026)
forecast_df['date'] = pd.to_datetime(forecast_df['year'].astype(str) + '-01-01')
forecast_df = forecast_df[forecast_df['Entity'] == selected_country]

# Checkbox untuk grafik kiri
show_predicted = st.checkbox("Tampilkan Predicted (2000–2024)", value=True)

# Layout 2 kolom (kiri: actual/predicted, kanan: forecast)
col1, col2 = st.columns(2)

# Grafik kiri: Actual & Predicted
with col1:
    st.subheader("📈 Data Historis (2000–2024)")
    fig1, ax1 = plt.subplots(figsize=(6, 4))

    ax1.plot(historical['date'], historical['Average surface temperature year'], 'bo-', label='Actual Temperature')

    if show_predicted:
        predicted = historical['Average surface temperature year'] + historical['Temperature anomaly']
        ax1.plot(historical['date'], predicted, 'rx--', label='Predicted Temperature')

    ax1.set_xlabel("Year")
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title("Actual & Predicted")
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

# Grafik kanan: Forecast
with col2:
    st.subheader("🔮 Forecast (2025–2026)")
    fig2, ax2 = plt.subplots(figsize=(6, 4))

    ax2.plot(forecast_df['date'], forecast_df['Forecast'], 'gD-', label='Forecasted Temperature')
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Temperature (°C)")
    ax2.set_title("Forecast")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)
