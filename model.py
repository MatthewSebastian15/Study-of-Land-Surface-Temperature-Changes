import pandas as pd
import numpy as np
from math import sqrt
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data_average_surface_temperature.csv")
df = df.dropna()
df = df[['Entity', 'year', 'Day', 'Average surface temperature month']]
df['month'] = pd.to_datetime(df['Day']).dt.month

def predict_temperature(entity_name):
    df_entity = df[df['Entity'] == entity_name].copy()
    df_entity = df_entity[['year', 'month', 'Average surface temperature month']]
    df_entity.columns = ['year', 'month', 'temperature']
    df_entity = df_entity.sort_values(['year', 'month']).reset_index(drop=True)

    df_entity['sin_month'] = np.sin(2 * np.pi * df_entity['month'] / 12)
    df_entity['cos_month'] = np.cos(2 * np.pi * df_entity['month'] / 12)

    df_entity['lag_1'] = df_entity['temperature'].shift(1)
    df_entity['lag_12'] = df_entity['temperature'].shift(12)
    df_entity['rolling_mean_3'] = df_entity['temperature'].rolling(window=3).mean()
    df_entity['rolling_std_3'] = df_entity['temperature'].rolling(window=3).std()
    df_entity['year_month'] = df_entity['year'] + (df_entity['month'] - 1) / 12

    df_entity = df_entity.dropna().reset_index(drop=True)

    feature_cols = ['year', 'sin_month', 'cos_month','lag_1', 'lag_12', 'rolling_mean_3', 'rolling_std_3', 'year_month']
    X = df_entity[feature_cols]
    y = df_entity['temperature']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train[['year']] = scaler.fit_transform(X_train[['year']])
    X_test[['year']] = scaler.transform(X_test[['year']])

    xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3,subsample=0.7, colsample_bytree=0.7, gamma=0.2, random_state=42)
    xgb_model.fit(X_train, y_train)

    y_pred_test = xgb_model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred_test))
    print(f"RMSE for {entity_name}: {rmse:.2f} °C")

    last_data = df_entity.copy()
    predictions = []

    start_date = datetime(2024, 7, 1)
    end_date = datetime(2030, 12, 31)
    current_date = start_date

    temp_mean = df_entity['temperature'].mean()
    temp_std = df_entity['temperature'].std()

    while current_date <= end_date:
        year = current_date.year
        month = current_date.month
        sin_month = np.sin(2 * np.pi * month / 12)
        cos_month = np.cos(2 * np.pi * month / 12)
        year_month = year + (month - 1) / 12

        lag_1 = last_data.iloc[-1]['temperature']
        lag_12 = last_data.iloc[-12]['temperature'] if len(last_data) >= 12 else lag_1
        rolling_mean_3 = last_data['temperature'].tail(3).mean()
        rolling_std_3 = last_data['temperature'].tail(3).std()

        year_scaled = scaler.transform([[year]])[0][0]

        features = np.array([[year_scaled, sin_month, cos_month,
                              lag_1, lag_12, rolling_mean_3, rolling_std_3, year_month]])
        xgb_pred = xgb_model.predict(features)[0]

        final_pred = np.clip(xgb_pred, temp_mean - 2*temp_std, temp_mean + 2*temp_std)

        new_row = {
            'year': year,
            'month': month,
            'temperature': final_pred,
            'sin_month': sin_month,
            'cos_month': cos_month,
            'lag_1': lag_1,
            'lag_12': lag_12,
            'rolling_mean_3': rolling_mean_3,
            'rolling_std_3': rolling_std_3,
            'year_month': year_month,
            'type': 'Predicted'
        }

        predictions.append(new_row)
        last_data = pd.concat([last_data, pd.DataFrame([new_row])], ignore_index=True)

        current_date = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)

    df_entity['type'] = 'Actual'
    result_df = pd.concat(
        [df_entity[['year', 'month', 'temperature', 'type']],
         pd.DataFrame(predictions)[['year', 'month', 'temperature', 'type']]],
        ignore_index=True
    )

    return result_df

def classify_climate_zone(result_df, entity_name):
    actual_df = result_df[result_df['type'] == 'Actual']
    annual_avg = actual_df.groupby('year')['temperature'].mean()
    overall_avg_temp = annual_avg.mean()

    if overall_avg_temp >= 20:
        climate_type = "Tropical"
    elif overall_avg_temp >= 10:
        climate_type = "Subtropical"
    elif overall_avg_temp >= 0:
        climate_type = "Temperate"
    else:
        climate_type = "Pole"

    return {
        "entity": entity_name,
        "average_annual_temperature": round(overall_avg_temp, 2),
        "climate_zone": climate_type
    }
