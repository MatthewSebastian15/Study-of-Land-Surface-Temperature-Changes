import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import predict_temperature

# Path ke file CSV
DATA_PATH = "data_average_surface_temperature.csv"

# Judul 
st.title("🌍 Surface Temperature Prediction")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Average surface temperature year'])
    df['date'] = pd.to_datetime(df['year'].astype(str) + '-01-01')
    return df

df = load_data()

# Sidebar: Pilih negara
st.sidebar.header("🔧 Pengaturan")
countries = sorted(df['Entity'].unique())
selected_country = st.sidebar.selectbox("Pilih Negara", countries)

# Filter berdasarkan negara
country_data = df[df['Entity'] == selected_country].copy()
historical = country_data[country_data['year'] < 2025]

# Prediksi forecast (untuk seluruh data, filter nanti)
forecast_df = predict_temperature(df, 2025, 2026)
forecast_df = forecast_df[forecast_df['Entity'] == selected_country]
forecast_df['date'] = pd.to_datetime(forecast_df['year'].astype(str) + '-01-01')

# Chart utama: Actual (dan opsi untuk predicted)
st.subheader(f"📊 Temperatur Aktual untuk {selected_country} (2000–2024)")

fig, ax = plt.subplots(figsize=(8, 5))

# Plot actual data
ax.plot(historical['date'], historical['Average surface temperature year'], 'bo-', label='Actual Temperature')

# Checkbox untuk predicted
show_predicted = st.checkbox("Tampilkan Predicted (2000–2024)", value=True)
if show_predicted and 'Temperature anomaly' in historical.columns:
    predicted = historical['Average surface temperature year'] + historical['Temperature anomaly']
    ax.plot(historical['date'], predicted, 'rx--', label='Predicted Temperature')

# Button untuk menampilkan forecast
show_forecast = st.button("Tampilkan Forecast 2025–2026")

# Tambahkan forecast jika tombol ditekan
if show_forecast:
    ax.plot(forecast_df['date'], forecast_df['Forecast'], 'gD-', label='Forecasted Temperature')

# Format chart
ax.set_xlabel("Year")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Surface Temperature")
ax.legend()
ax.grid(True)

# Tampilkan chart
st.pyplot(fig)
