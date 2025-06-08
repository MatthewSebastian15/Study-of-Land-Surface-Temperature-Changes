import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Data CSV berada di root project (pastikan namanya sesuai file yang Anda upload)
DATA_PATH = "data_average_surface_temperature.csv"

# Judul Aplikasi
st.title("🌡️ Surface Temperature Prediction: Historical and Future")

# Load data dengan cache
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Cek apakah kolom year dan month ada
    if 'year' not in df.columns or 'month' not in df.columns:
        st.error("Kolom 'year' dan 'month' tidak ditemukan dalam data.")
        st.stop()

    # Buat kolom tanggal (datetime)
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

    return df

df = load_data()

# Pilih negara
countries = df['country'].unique()
selected_country = st.selectbox("Select Country", sorted(countries))

# Filter data berdasarkan negara yang dipilih
country_data = df[df['country'] == selected_country]

# Bagi data jadi historis dan prediksi masa depan
historical = country_data[country_data['date'] < '2025']
future = country_data[country_data['date'] >= '2025']

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot data aktual
if 'actual_temp' in historical.columns:
    ax.plot(historical['date'], historical['actual_temp'], 'bo-', label='Actual Temp (2000–2024)')

# Plot prediksi historis
if 'predicted_temp' in historical.columns:
    ax.plot(historical['date'], historical['predicted_temp'], 'rx--', label='Predicted Temp (2000–2024)')

# Plot prediksi masa depan jika tersedia
if not future.empty and 'future_predicted_temp' in future.columns:
    ax.plot(future['date'], future['future_predicted_temp'], 'gD-', label='Future Prediction (2025–2026)')

# Keterangan & Label
ax.set_title(f"Temperature Forecast: {selected_country}")
ax.set_xlabel("Time")
ax.set_ylabel("Temperature (°C)")
ax.grid(True)
ax.legend()

st.pyplot(fig)
