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
    df.columns = df.columns.str.strip()  # Hapus spasi ekstra dari nama kolom
    df['date'] = pd.to_datetime(df['year'].astype(str) + '-01-01')
    return df

df = load_data()

# Pilihan negara
countries = sorted(df['Entity'].unique())
selected_country = st.selectbox("Pilih Negara", countries)

# Filter berdasarkan negara
country_data = df[df['Entity'] == selected_country]

# Checkbox untuk jenis data yang ingin ditampilkan
show_actual = st.checkbox("Show Actual Temperature (2000–2024)", value=True)
show_predicted = st.checkbox("Show Predicted Temperature (2000–2024)", value=True)
show_forecast = st.checkbox("Show Forecast (2025–2026)", value=False)

# Filter data historis
historical = country_data[country_data['year'] < 2025]

# Prediksi masa depan (forecast)
future_years = [2025, 2026]
if show_forecast:
    forecast_df = predict_temperature(country_data, 2025, 2026)
    forecast_df['date'] = pd.to_datetime(forecast_df['year'].astype(str) + "-01-01")
    forecast_df = forecast_df[forecast_df['Entity'] == selected_country]

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

if show_actual:
    ax.plot(historical['date'], historical['Average surface temperature year'], 'bo-', label='Actual Temperature (2000–2024)')

if show_predicted:
    predicted = historical['Average surface temperature year'] + historical['Temperature anomaly']
    ax.plot(historical['date'], predicted, 'rx--', label='Predicted Temperature (2000–2024)')

if show_forecast and not forecast_df.empty:
    ax.plot(forecast_df['date'], forecast_df['Forecast'], 'gD-', label='Future Predicted Temperature (2025–2026)')

ax.set_title(f"Surface Temperature Prediction for {selected_country}")
ax.set_xlabel("Year")
ax.set_ylabel("Surface Temperature (°C)")
ax.grid(True)
ax.legend()

st.pyplot(fig)
