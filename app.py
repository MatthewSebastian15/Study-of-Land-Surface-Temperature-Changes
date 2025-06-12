import streamlit as st
import pandas as pd
import plotly.express as px
from model import predict_temperature, classify_climate_zone

st.set_page_config(layout="wide", page_title="Surface Temperature Prediction 🌍")

st.markdown(
    "<h1 style='text-align: center; color: #00BFFF;'>🌡️ Surface Temperature Forecast</h1>",
    unsafe_allow_html=True
)

df_all = pd.read_csv("data_average_surface_temperature.csv")
entities = sorted(df_all['Entity'].dropna().unique())

with st.sidebar:
    st.markdown("## 🌍 Select Country")
    default_index = entities.index("Indonesia") if "Indonesia" in entities else 0
    selected_entity = st.selectbox("", entities, index=default_index)
    display_actual = st.checkbox("📊 Show Historical Average (2000 – 2024)", value=True)
    display_predicted = st.checkbox("📈 Show Monthly Forecast (2025 – 2030)", value=True)

result_df = predict_temperature(selected_entity)
result_df['date'] = pd.to_datetime(result_df['year'].astype(str) + '-' + result_df['month'].astype(str))

plot_df = pd.DataFrame()

if display_actual:
    actual_df = result_df[result_df['type'] == 'Actual']
    actual_df = actual_df[(actual_df['date'] >= '1974-01-01') & (actual_df['date'] <= '2024-08-31')]
    if not actual_df.empty:
        date_filter = pd.date_range(start='1974-01-01', end='2024-08-01', freq='6MS')
        monthly_actual = actual_df[actual_df['date'].isin(date_filter)]
        monthly_actual = monthly_actual[['date', 'temperature', 'type']]
        monthly_actual.rename(columns={'temperature': 'Temperature (°C)'}, inplace=True)
        plot_df = pd.concat([plot_df, monthly_actual], ignore_index=True)

if display_predicted:
    result_df['date'] = pd.to_datetime(result_df['year'].astype(str) + '-' + result_df['month'].astype(str) + '-01')
    predicted_df = result_df[(result_df['type'] == 'Predicted') & (result_df['date'].between('2024-07-01', '2030-12-31'))]
    if not predicted_df.empty:
        predicted_df = predicted_df[['date', 'temperature', 'type']]
        predicted_df.rename(columns={'temperature': 'Temperature (°C)'}, inplace=True)
        plot_df = pd.concat([plot_df, predicted_df], ignore_index=True)

if not plot_df.empty:
    fig = px.line(
        plot_df, x='date', y='Temperature (°C)',
        color='type',
        labels={'Temperature (°C)': 'Temperature (°C)', 'date': 'Date', 'type': 'Data type'},
        color_discrete_map={'Actual': '#00BFFF', 'Predicted': '#FF4500'},
        title=f"📈 Average Surface Temperature & Predictions {selected_entity}",
    )

    fig.update_traces(
        hovertemplate='📅 %{x|%b %Y}<br>🌡️ %{y:.2f}°C<extra></extra>',
        line=dict(width=2, shape='spline')
    )

    fig_width = 1400 if display_predicted and not display_actual else 1000

    fig.update_layout(
        template='plotly_dark',
        margin=dict(l=80, r=20, t=50, b=50),
        height=500,
        width=fig_width,
        font=dict(family="Arial", size=14),
        title=dict(font=dict(size=22)),
        legend=dict(font=dict(size=14), orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    if display_predicted and not display_actual:
        x_start, x_end = pd.to_datetime("2024-07-01"), pd.to_datetime("2030-12-31")
        tickvals = pd.date_range(start=x_start, end=x_end, freq='MS')
    elif display_actual and display_predicted:
        x_start, x_end = pd.to_datetime("1974-01-01"), pd.to_datetime("2030-12-31")
        tickvals = pd.date_range(start=f"{x_start.year}-01-01", end=f"{x_end.year}-12-01", freq='2YS')
    else:
        x_start, x_end = pd.to_datetime("1974-01-01"), pd.to_datetime("2025-01-01")
        tickvals = pd.date_range(start=f"{x_start.year}-01-01", end=f"{x_end.year}-12-01", freq='2YS')

    fig.update_xaxes(
        range=[x_start, x_end],
        tickvals=tickvals[::2] if display_predicted and not display_actual else tickvals,
        tickformat="%b %Y" if display_predicted and not display_actual else "%Y",
        tickangle=45,
        tickfont=dict(size=11),
        ticklabelmode='instant',
        automargin=True,
        gridcolor='gray',
        gridwidth=0.5
    )

    st.plotly_chart(fig, use_container_width=False)

    with st.expander("📌 Climate Classification Summary", expanded=True):
        climate_info = classify_climate_zone(result_df, selected_entity)
        st.markdown(
            f"""
            <div style='background-color:#002b36;padding:15px;border-radius:10px'>
                <h4 style='color:#00BFFF'>🌡️ {climate_info['entity']} is included in the climate zone</h4>
                <ul style='color:white;font-size:16px'>
                    <li>Countries with <strong>{climate_info['climate_zone']}</strong> climates</li>
                    <li><strong>Average Annual Temperature (2000–2024) : </strong> {climate_info['average_annual_temperature']} °C</li>
                </ul>
                <p style='font-size:14px;color:gray'>*Classification based on average annual temperature from historical data.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

else:
    st.warning("⚠️ Charts are not available")
