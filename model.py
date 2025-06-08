import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

continent_encoder = LabelEncoder()
entity_encoder = LabelEncoder()

def train_model(df):
    df = df[['Entity', 'Continent', 'year', 'Average surface temperature year']].dropna()
    df['Continent_enc'] = continent_encoder.fit_transform(df['Continent'])
    df['Entity_enc'] = entity_encoder.fit_transform(df['Entity'])
    X = df[['Entity_enc', 'Continent_enc', 'year']]
    y = df['Average surface temperature year']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"Validation RMSE : {rmse:.4f}")
    return model

def predict_temperature(df, start_year=2025, end_year=2026):
    model = train_model(df)
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

    try:
        forecast_df['Continent_enc'] = continent_encoder.transform(forecast_df['Continent'])
        forecast_df['Entity_enc'] = entity_encoder.transform(forecast_df['Entity'])
    except ValueError as e:
        raise ValueError("Entity or Continent in forecast not seen in training data.") from e
    X_forecast = forecast_df[['Entity_enc', 'Continent_enc', 'year']]
    forecast_df['Forecast'] = model.predict(X_forecast)
    return forecast_df[['Entity', 'Continent', 'year', 'Forecast']]
