import streamlit as st
import pandas as pd
import calendar
import time
import plotly.graph_objects as go
from model import predict_temperature, classify_climate_zone

st.set_page_config(layout="wide", page_title="🌡️ Temperature Forecast")
st.markdown("""<h1 style='text-align: center; color: #00BFFF;'>🌡️ Surface Temperature Forecast</h1>""", unsafe_allow_html=True)

df_all = pd.read_csv("data_average_surface_temperature.csv")
entities = sorted(df_all['Entity'].dropna().unique())

st.markdown("---")

with st.container():
    st.markdown("### 🌍 Select Country")
    col1, col2, col3 = st.columns([5, 2.2, 2.2])

    with col1:
        default_index = entities.index("Indonesia") if "Indonesia" in entities else 0
        selected_entity = st.selectbox(" ", entities, index=default_index)

    with col2:
        display_actual = st.checkbox("📊 Historical", value=True)
        if display_actual:
            actual_range = st.slider("", min_value=1940, max_value=2024, value=(2000, 2024), key="actual", label_visibility="collapsed")

    with col3:
        display_predicted = st.checkbox("📈 Forecast", value=True)
        if display_predicted:
            predict_range = st.slider("", min_value=2025, max_value=2030, value=(2025, 2030), key="predict", label_visibility="collapsed")

st.markdown("---")

data = predict_temperature(selected_entity)
data['date'] = pd.to_datetime(data['year'].astype(str) + '-' + data['month'].astype(str) + '-01')

chart_placeholder = st.empty()
with st.spinner("⏳ Loading chart..."):
    time.sleep(1)

fig = go.Figure()

if display_actual:
    actual = data[(data['type'] == 'Actual') & (data['year'].between(*actual_range))]
    if not actual.empty:
        actual['Month'] = actual['month'].apply(lambda x: calendar.month_name[x])
        actual['Country'] = selected_entity
        customdata = actual[['year', 'Month', 'Country']].values

        fig.add_trace(go.Scatter(
            x=actual['date'],
            y=actual['temperature'],
            mode='lines',
            name='Actual',
            line=dict(color='#00BFFF', width=2),
            customdata=customdata,
            hovertemplate='<b>%{customdata[2]}</b><br>🗓️ Year: %{customdata[0]}<br>🗓️ Month: %{customdata[1]}<br>🌡️ Temp: %{y:.2f} °C<extra></extra>'
        ))

if display_predicted:
    predicted = data[(data['type'] == 'Predicted') & (data['year'].between(*predict_range))]
    if not predicted.empty:
        predicted['Month'] = predicted['month'].apply(lambda x: calendar.month_name[x])
        predicted['Country'] = selected_entity
        customdata = predicted[['year', 'Month', 'Country']].values

        fig.add_trace(go.Scatter(
            x=predicted['date'],
            y=predicted['temperature'],
            mode='lines',
            name='Predicted',
            line=dict(color='#FF4500', width=2),
            customdata=customdata,
            hovertemplate='<b>%{customdata[2]}</b><br>🗓️ Year: %{customdata[0]}<br>🗓️ Month: %{customdata[1]}<br>🌡️ Temp: %{y:.2f} °C<extra></extra>'
        ))

fig.update_layout(
    title=f"📈 {selected_entity} Surface Temperature Over Time",
    template='plotly_dark',
    margin=dict(l=80, r=20, t=50, b=50),
    height=500,
    width=1300,
    font=dict(family="Arial", size=14),
    title_font=dict(size=22),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    hovermode='x',
    xaxis=dict(showline=True, showgrid=True, gridcolor='rgba(255, 255, 255, 0.08)', tickformat="%Y"),
    yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.08)')
)

if fig.data:
    chart_placeholder.plotly_chart(fig, use_container_width=True)
    climate_info = classify_climate_zone(data, selected_entity)
    continent_name = df_all[df_all['Entity'] == selected_entity]['Continent'].dropna().unique()
    continent_name = continent_name[0] if len(continent_name) > 0 else "n/a"

    actual_avg = data[(data['type'] == 'Actual') & (data['year'].between(2000, 2024))]['temperature'].mean()
    predicted_avg = data[(data['type'] == 'Predicted') & (data['year'].between(2025, 2030))]['temperature'].mean()

    actual_avg_str = f"{actual_avg:.2f} °C" if not pd.isna(actual_avg) else "n/a"
    predicted_avg_str = f"{predicted_avg:.2f} °C" if not pd.isna(predicted_avg) else "n/a"
    climate_zone = climate_info['climate_zone'] if climate_info['climate_zone'] else "n/a"

    if not pd.isna(actual_avg) and not pd.isna(predicted_avg):
        delta = predicted_avg - actual_avg
        delta_pct = (delta / actual_avg) * 100
        if delta > 0:
            trend_icon = "↑"
            trend_color = "#00FF00"
        elif delta < 0:
            trend_icon = "↓"
            trend_color = "#FF6347"
        else:
            trend_icon = "➖"
            trend_color = "gray"
        trend_value = f"{trend_icon} {delta:+.2f} °C"
        trend_percent = f"({delta_pct:+.1f}%)"
    else:
        trend_value = "–"
        trend_percent = "n/a"
        trend_color = "gray"

    st.markdown(f"<h3 style='color:#00BFFF; text-align:center; margin-bottom:10px;'>🌍 {selected_entity} — Climate Overview</h3><hr style='border:1px solid gray; margin-bottom:30px;'>", unsafe_allow_html=True)
    cols = st.columns(5)

    with cols[0]:
        st.markdown(f"""
        <div style='text-align:center;'>
            <div style='font-size:12px;color:#cccccc;'>🌐 Continent</div>
            <div style='font-size:28px;font-weight:bold;color:white;'>{continent_name}</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[1]:
        st.markdown(f"""
        <div style='text-align:center;'>
            <div style='font-size:12px;color:#cccccc;'>🌾 Climate Zone</div>
            <div style='font-size:28px;font-weight:bold;color:white;'>{climate_zone}</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[2]:
        st.markdown(f"""
        <div style='text-align:center;'>
            <div style='font-size:12px;color:#cccccc;'>🗕 Historical Avg</div>
            <div style='font-size:28px;font-weight:bold;color:white;'>{actual_avg_str}</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[3]:
        st.markdown(f"""
        <div style='text-align:center;'>
            <div style='font-size:12px;color:#cccccc;'>🔮 Predicted Avg</div>
            <div style='font-size:28px;font-weight:bold;color:white;'>{predicted_avg_str}</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[4]:
        st.markdown(f"""
        <div style='text-align:center;'>
            <div style='font-size:12px;color:#cccccc;'>📈 Trend</div>
            <div style='font-size:28px;font-weight:bold;color:{trend_color};'>
                {trend_value}
            </div>
            <div style='font-size:16px;color:{trend_color};'>
                {trend_percent}
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.warning("⚠️ No data available to display the chart")
