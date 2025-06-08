import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Path ke file data Anda (ubah sesuai lokasi sebenarnya)
DATA_PATH = "data_average_surface_temperature.csv"

# Judul Aplikasi
st.title("Surface Temperature Prediction: Historical and Future")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    return df

df = load_data()

# Dropdown negara
countries = df['country'].unique()
selected_country = st.selectbox("Select Country", sorted(countries))

# Filter data berdasarkan negara
country_data = df[df['country'] == selected_country]

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

# Data historis
historical = country_data[country_data['date'] < '2025']
future = country_data[country_data['date'] >= '2025']

# Plot data aktual
ax.plot(historical['date'], historical['actual_temp'], 'bo-', label='Actual Temperature (2000–2024)')
# Plot prediksi historis
ax.plot(historical['date'], historical['predicted_temp'], 'rx--', label='Predicted Temperature (2000–2024)')
# Plot prediksi masa depan
if not future.empty and 'future_predicted_temp' in future.columns:
    ax.plot(future['date'], future['future_predicted_temp'], 'gD-', label='Future Predicted Temperature (2025–2026)')

ax.set_title(f"Surface Temperature Prediction: {selected_country}")
ax.set_xlabel("Time")
ax.set_ylabel("Surface Temperature (°C)")
ax.legend()
ax.grid(True)

st.pyplot(fig)
