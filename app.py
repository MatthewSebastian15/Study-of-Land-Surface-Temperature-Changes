import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Path ke file CSV (lokal atau relatif ke folder proyek)
DATA_PATH = "data_average_surface_temperature.csv"

# Judul Aplikasi
st.title("Surface Temperature Prediction: Historical and Future")

# Load data dengan cache
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    # Pastikan 'year' dan 'month' ada
    if 'year' in df.columns and 'month' in df.columns:
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    else:
        st.error("Kolom 'year' dan 'month' tidak ditemukan di dataset.")
    return df

# Load data
df = load_data()

# Cek jika kolom negara ada
if 'country_name' in df.columns:
    countries = df['country_name'].unique()
    selected_country = st.selectbox("Pilih Negara", sorted(countries))

    # Filter berdasarkan negara
    country_data = df[df['country_name'] == selected_country]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    historical = country_data[country_data['date'] < '2025']
    future = country_data[country_data['date'] >= '2025']

    # Plot temperatur aktual
    if 'actual_temp' in df.columns:
        ax.plot(historical['date'], historical['actual_temp'], 'bo-', label='Actual Temp (2000–2024)')
    
    # Plot prediksi masa lalu
    if 'predicted_temp' in df.columns:
        ax.plot(historical['date'], historical['predicted_temp'], 'rx--', label='Predicted Temp (2000–2024)')

    # Plot prediksi masa depan
    if 'future_predicted_temp' in df.columns and not future.empty:
        ax.plot(future['date'], future['future_predicted_temp'], 'gD-', label='Future Predicted Temp (2025–2026)')

    ax.set_title(f"Surface Temperature Prediction: {selected_country}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Surface Temperature (°C)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

else:
    st.error("Kolom 'country_name' tidak ditemukan di dataset.")
