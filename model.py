import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

# Inisialisasi encoder global
continent_encoder = LabelEncoder()
entity_encoder = LabelEncoder()

def train_model(df):
    # Pastikan kolom yang dibutuhkan tersedia
    df = df[['Entity', 'Continent', 'year', 'Average surface temperature year']].dropna()

    # Encode hanya nilai unik yang ada
    df['Continent_enc'] = continent_encoder.fit_transform(df['Continent'])
    df['Entity_enc'] = entity_encoder.fit_transform(df['Entity'])

    X = df[['Entity_enc', 'Continent_enc', 'year']]
    y = df['Average surface temperature year']

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    return model

def predict_temperature(df, start_year=2025, end_year=2026):
    model = train_model(df)

    # Ambil kombinasi unik Entity dan Continent dari data
    unique_entities = df[['Entity', 'Continent']].drop_duplicates()

    forecast_rows = []
    for _, row in unique_entities.iterrows():
        entity = row['Entity']
        continent = row['Continent']
        for year in range(start_year, end_year + 1):
            forecast_rows.append({
                'Entity': entity,
                'Continent': continent,
                'year': year
            })

    forecast_df = pd.DataFrame(forecast_rows)

    # Validasi agar tidak terjadi KeyError
    try:
        forecast_df['Continent_enc'] = continent_encoder.transform(forecast_df['Continent'])
        forecast_df['Entity_enc'] = entity_encoder.transform(forecast_df['Entity'])
    except ValueError:
        raise ValueError("Ada nilai Entity atau Continent pada prediksi yang belum pernah dilihat saat training.")

    X_forecast = forecast_df[['Entity_enc', 'Continent_enc', 'year']]
    forecast_df['Forecast'] = model.predict(X_forecast)

    return forecast_df[['Entity', 'Continent', 'year', 'Forecast']]
