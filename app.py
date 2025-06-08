import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import predict_temperature

# File CSV
DATA_PATH = "data_average_surface_temperature.csv"

# Judul
st.title("Surface Temperature Prediction: Historical and Future")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['year'].astype(str) + '-01-01')
    return df

df = load_data()

# Prediksi suhu menggunakan model
df = predict_temperature(df)

# Pilihan negara
countries = df['Entity'].unique()
selected_country = st.selectbox("Pilih Negara", sorted(countries))

# Filter negara
country_data = df[df['Entity'] == selected_country]

# Pisahkan data
historical = country_data[country_data['year'] < 2025]
forecast = country_data[country_data['year'] >= 2025]

# Checkbox untuk kontrol plot
show_actual = st.checkbox("Tampilkan Data Aktual (2000–2024)", value=True)
show_predicted = st.checkbox("Tampilkan Prediksi (2000–2024)", value=False)
show_forecast = st.checkbox("Tampilkan Forecast Masa Depan (2025–2026)", value=False)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

if show_actual:
    ax.plot(historical['date'], historical['Average surface temperature year'], 'bo-', label='Actual Temperature (2000–2024)')

if show_predicted and 'predicted_temp' in historical.columns:
    ax.plot(historical['date'], historical['predicted_temp'], 'rx--', label='Predicted Temperature (2000–2024)')

if show_forecast and not forecast.empty and 'future_predicted_temp' in forecast.columns:
    ax.plot(forecast['date'], forecast['future_predicted_temp'], 'gD-', label='Future Predicted Temperature (2025–2026)')

ax.set_title(f"Surface Temperature Prediction: Historical and Future ({selected_country})")
ax.set_xlabel("Time")
ax.set_ylabel("Surface Temperature (°C)")
ax.legend()
ax.grid(True)

st.pyplot(fig)
