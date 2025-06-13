import streamlit as st
import pandas as pd
import plotly.express as px
from model import predict_temperature, classify_climate_zone

st.set_page_config(layout="wide", page_title="Surface Temperature Prediction 🌍")

st.markdown("<h1 style='text-align: center; color: #00BFFF;'>🌡️ Surface Temperature Forecast</h1>",unsafe_allow_html=True)

# Load data
df_all = pd.read_csv("dataset/data_average_surface_temperature.csv")
entities = sorted(df_all['Entity'].dropna().unique())

# Sidebar UI
with st.sidebar:
    st.markdown("## 🌍 Select Country")
    default_index = entities.index("Indonesia") if "Indonesia" in entities else 0
    selected_entity = st.selectbox("", entities, index=default_index)
    display_actual = st.checkbox("📊 Show Historical Average (2000 – 2024)", value=True)
    display_predicted = st.checkbox("📈 Show Monthly Forecast (2025 – 2030)", value=True)

# Predict and preprocess
data = predict_temperature(selected_entity)
data['date'] = pd.to_datetime(data['year'].astype(str) + '-' + data['month'].astype(str) + '-01')
plot_df = pd.DataFrame()

# Actual data
if display_actual:
    actual = data[(data['type'] == 'Actual') & (data['date'].between('1974-01-01', '2024-08-31'))]
    actual = actual[actual['date'].dt.month.isin([1, 7])]
    actual = actual[['date', 'temperature', 'type']].rename(columns={'temperature': 'Temperature (°C)'})
    plot_df = pd.concat([plot_df, actual], ignore_index=True)

# Predicted data
if display_predicted:
    predicted = data[(data['type'] == 'Predicted') & (data['date'].between('2024-07-01', '2030-12-31'))]
    predicted = predicted[['date', 'temperature', 'type']].rename(columns={'temperature': 'Temperature (°C)'})
    plot_df = pd.concat([plot_df, predicted], ignore_index=True)

# Chart
if not plot_df.empty:
    fig = px.line(
        plot_df, x='date', y='Temperature (°C)', color='type',
        labels={'Temperature (°C)': 'Temperature (°C)', 'date': 'Date', 'type': 'Data type'},
        color_discrete_map={'Actual': '#00BFFF', 'Predicted': '#FF4500'},
        title=f"📈 {selected_entity} Surface Temperature Overview"
    )
    fig.update_traces(
        hovertemplate='📅 %{x|%b %Y}<br>🌡️ %{y:.2f}°C<extra></extra>',
        line=dict(width=2, shape='spline')
    )

    # X-axis tick handling
    if display_actual and display_predicted:
        x_start, x_end = pd.to_datetime("1974-01-01"), pd.to_datetime("2030-12-31")
        tickvals = pd.date_range(x_start, x_end, freq='2YS')
    elif display_predicted:
        x_start, x_end = pd.to_datetime("2024-07-01"), pd.to_datetime("2030-12-31")
        tickvals = pd.date_range(x_start, x_end, freq='MS')[::2]
    else:
        x_start, x_end = pd.to_datetime("1974-01-01"), pd.to_datetime("2025-01-01")
        tickvals = pd.date_range(x_start, x_end, freq='2YS')

    fig.update_layout(
        template='plotly_dark',
        margin=dict(l=80, r=20, t=50, b=50),
        height=500,
        width=1300,
        font=dict(family="Arial", size=14),
        title=dict(font=dict(size=22)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    fig.update_xaxes(
        range=[x_start, x_end],
        tickvals=tickvals,
        tickformat="%b %Y" if display_predicted and not display_actual else "%Y",
        tickangle=45,
        tickfont=dict(size=11),
        automargin=True,
        gridcolor='gray',
        gridwidth=0.5
    )
    st.plotly_chart(fig, use_container_width=True)

    # Climate Summary
    with st.expander("📌 Climate Classification Summary", expanded=True):
        climate_info = classify_climate_zone(data, selected_entity)

        # Ambil nama benua
        continent_name = df_all[df_all['Entity'] == selected_entity]['Continent'].dropna().unique()
        continent_name = continent_name[0] if len(continent_name) > 0 else "Unknown"

        # Hitung suhu rata-rata aktual dan prediksi
        actual_avg = data[(data['type'] == 'Actual') & (data['year'].between(2000, 2024))]['temperature'].mean()
        predicted_avg = data[(data['type'] == 'Predicted') & (data['year'].between(2025, 2030))]['temperature'].mean()

        actual_avg_str = f"{actual_avg:.2f} °C" if not pd.isna(actual_avg) else "N/A"
        predicted_avg_str = f"{predicted_avg:.2f} °C" if not pd.isna(predicted_avg) else "N/A"

        # Hitung tren
        if not pd.isna(actual_avg) and not pd.isna(predicted_avg):
            delta = predicted_avg - actual_avg
            delta_pct = (delta / actual_avg) * 100
            if delta > 0:
                trend_icon = "⬆"
                trend_color = "#32CD32"  
            elif delta < 0:
                trend_icon = "⬇"
                trend_color = "#FF6347"  
            else:
                trend_icon = "➖"
                trend_color = "gray"
            trend_text = f"<span style='color:{trend_color}'>{trend_icon} {delta:+.2f} °C ({delta_pct:+.1f}%)</span>"
        else:
            trend_text = "<span style='color:gray'>N/A</span>"

        # Tampilkan ke UI
        st.markdown(f"""
            <div style='background-color:#001f3f;padding:20px;border-radius:12px;border: 1px solid #00BFFF'>
                <h3 style='color:#00BFFF;text-align:center;'>🌍 {selected_entity} — Climate Overview</h3>
                <hr style='border-top: 1px solid #00BFFF;'/>
                <p style='color:white;font-size:16px'>
                    <strong>🌐 Continent </strong> {continent_name}<br>
                    <strong>🏜️ Climate Zone </strong> {climate_info['climate_zone']}<br>
                </p>
                <p style='color:white;font-size:16px'>
                    <strong>📅 Historical Avg (2000–2024) </strong> {actual_avg_str}<br>
                    <strong>🔮 Predicted Avg (2025–2030) </strong> {predicted_avg_str}<br>
                    <strong>📈 Trend </strong> {trend_text}
                </p>
                <p style='font-size:13px;color:gray'>*Climate classification is based on long-term average surface temperatures</p>
            </div>
        """, unsafe_allow_html=True)
else:
    st.warning("⚠️ No data available to display the chart")
