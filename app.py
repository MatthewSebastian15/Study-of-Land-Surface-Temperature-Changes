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
    df.columns = df.columns.str.strip()  # Bersihkan nama kolom
    df['date'] = pd.to_datetime(df['year'].astype(str) + '-01-01')
    return df

df = load_data()

# Pilih negara
countries = df['Entity'].unique()
selected_country = st.selectbox("Select Country", sorted(countries))

# Filter berdasarkan negara
country_data = df[df['Entity'] == selected_country]

# Buat prediksi masa depan
future_years = [2025, 2026]
future_dates = pd.to_datetime([f"{year}-01-01" for year in future_years])
future_preds = predict_temperature(country_data, future_years)
future_df = pd.DataFrame({
    'date': future_dates,
    'Future Predicted Temperature': future_preds
})

# Checkbox tampilan
show_actual = st.checkbox("Show Actual Temperature (2000–2024)", value=True)
show_predicted = st.checkbox("Show Predicted Temperature (2000–2024)", value=True)
show_forecast = st.checkbox("Show Forecast (2025–2026)", value=False)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
historical = country_data[country_data['year'] < 2025]

if show_actual:
    ax.plot(historical['date'], historical['Average surface temperature year'], 'bo-', label='Actual Temperature (2000–2024)')

if show_predicted:
    predicted = historical['Average surface temperature year'] + historical['Temperature anomaly']
    ax.plot(historical['date'], predicted, 'rx--', label='Predicted Temperature (2000–2024)')

if show_forecast:
    ax.plot(future_df['date'], future_df['Future Predicted Temperature'], 'gD-', label='Future Predicted Temperature (2025–2026)')

ax.set_title(f"Surface Temperature Prediction: Historical and Future ({selected_country})")
ax.set_xlabel("Time")
ax.set_ylabel("Surface Temperature (°C)")
ax.legend()
ax.grid(True)

st.pyplot(fig)
