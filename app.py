import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Gunakan path relatif
DATA_PATH = "data_average_surface_temperature.csv"

# Judul Aplikasi
st.title("Surface Temperature Prediction: Historical and Future")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Pastikan kolom 'year' dan 'month' ada
    if 'year' in df.columns and 'month' in df.columns:
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    else:
        st.error("Kolom 'year' dan 'month' tidak ditemukan di dataset.")
    return df

df = load_data()

# Dropdown negara
if 'country' in df.columns:
    countries = df['country'].unique()
    selected_country = st.selectbox("Select Country", sorted(countries))

    # Filter data berdasarkan negara
    country_data = df[df['country'] == selected_country]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    historical = country_data[country_data['date'] < '2025']
    future = country_data[country_data['date'] >= '2025']

    # Plot aktual & prediksi historis
    ax.plot(historical['date'], historical['actual_temp'], 'bo-', label='Actual Temp (2000–2024)')
    ax.plot(historical['date'], historical['predicted_temp'], 'rx--', label='Predicted Temp (2000–2024)')

    # Plot prediksi masa depan jika tersedia
    if not future.empty and 'future_predicted_temp' in future.columns:
        ax.plot(future['date'], future['future_predicted_temp'], 'gD-', label='Future Predicted Temp (2025–2026)')

    ax.set_title(f"Surface Temperature Prediction: {selected_country}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Surface Temperature (°C)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
else:
    st.error("Kolom 'country' tidak ditemukan di dataset.")
