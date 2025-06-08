import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

# Inisialisasi LabelEncoder global (untuk konsistensi encoding)
continent_encoder = LabelEncoder()
entity_encoder = LabelEncoder()

def train_model(df):
    # Hanya kolom yang dibutuhkan
    df = df[['Entity', 'Continent', 'year', 'Average surface temperature year']].dropna()

    # Encoding
    df['Continent'] = continent_encoder.fit_transform(df['Continent'])
    df['Entity'] = entity_encoder.fit_transform(df['Entity'])

    X = df[['Entity', 'Continent', 'year']]
    y = df['Average surface temperature year']

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    return model

def predict_temperature(df, start_year=2025, end_year=2026):
    model = train_model(df)

    all_forecasts = []
    unique_entities = df[['Entity', 'Continent']].drop_duplicates()

    for _, row in unique_entities.iterrows():
        entity = row['Entity']
        continent = row['Continent']
        for year in range(start_year, end_year + 1):
            all_forecasts.append({
                'Entity': entity,
                'Continent': continent,
                'year': year
            })

    forecast_df = pd.DataFrame(all_forecasts)

    # Encoding sama seperti saat training
    forecast_df['Continent'] = continent_encoder.transform(forecast_df['Continent'])
    forecast_df['Entity'] = entity_encoder.transform(forecast_df['Entity'])

    X_forecast = forecast_df[['Entity', 'Continent', 'year']]
    forecast_df['Forecast'] = model.predict(X_forecast)

    # Decode kembali label Entity dan Continent
    forecast_df['Entity'] = entity_encoder.inverse_transform(forecast_df['Entity'])
    forecast_df['Continent'] = continent_encoder.inverse_transform(forecast_df['Continent'])

    return forecast_df
