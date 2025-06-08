import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Path dataset
DATA_PATH = "data_average_surface_temperature.csv"

# Judul aplikasi
st.title("Surface Temperature Prediction: Historical and Future")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()  # Bersihkan spasi
    # Buat kolom 'date' dari 'year' saja (karena tidak ada bulan)
    df['date'] = pd.to_datetime(df['year'], format='%Y')
    return df

# Load data
df = load_data()

# Dropdown negara
if 'Entity' in df.columns:
    countries = df['Entity'].unique()
    selected_country = st.selectbox("Pilih Negara", sorted(countries))

    # Filter berdasarkan negara
    country_data = df[df['Entity'] == selected_country]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Data historis dan masa depan
    historical = country_data[country_data['year'] < 2025]
    future = country_data[country_data['year'] >= 2025]

    # Plot temperatur aktual (gunakan rata-rata tahunan)
    if 'Average surface temperature year' in df.columns:
        ax.plot(historical['date'], historical['Average surface temperature year'], 'bo-', label='Avg Temp (Historis)')
        if not future.empty:
            ax.plot(future['date'], future['Average surface temperature year'], 'gD-', label='Avg Temp (Future)')

    ax.set_title(f"Surface Temperature - {selected_country}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Avg Surface Temp (°C)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
else:
    st.error("Kolom 'Entity' tidak ditemukan di dataset.")
