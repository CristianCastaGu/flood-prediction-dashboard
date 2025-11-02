import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# =========================================================
# Configuración de la página
# =========================================================
st.set_page_config(
    page_title="Predicción de Inundaciones",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CSS personalizado para mejor diseño

# =========================================================
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #e0f2fe 0%, #e0f7fa 100%);
    }
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    div[data-testid="metric-container"] label {
        color: white !important;
        font-weight: bold;
    }
    h1 {
        color: #0284c7;
        text-align: center;
        font-size: 3em;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# Función mejorada para calcular riesgo de inundación
# =========================================================
def calculate_flood_risk(pressure, humidity, precipitation, flow_velocity, 
                         wind_speed, temperature, sediments, flow_volume):
    """
    Calcula el riesgo de inundación basado en parámetros hidrometeorológicos.
    Retorna un valor entre 0 y 100.
    """
    
    # Normalizar valores a escala 0-1
    norm_precipitation = min(precipitation / 200, 1)
    norm_humidity = humidity / 100
    norm_flow_volume = min(flow_volume / 2000, 1)
    norm_flow_velocity = min(flow_velocity / 10, 1)
    norm_sediments = min(sediments / 500, 1)
    
    # Presión baja aumenta riesgo (invertir escala)
    norm_pressure = 1 - max(0, min((pressure - 980) / 40, 1))
    norm_wind_speed = min(wind_speed / 80, 1)
    
    # Temperatura óptima para tormentas (26-30 grados tiene mayor riesgo)
    if 26 <= temperature <= 32:
        temp_risk = 0.8
    elif temperature > 32:
        temp_risk = 0.5
    else:
        temp_risk = max(0, (temperature - 15) / 15)
    
    # Pesos basados en importancia hidrológica real
    weights = {
        'precipitation': 0.30,      # Factor más crítico
        'flow_volume': 0.25,        # Volumen del caudal
        'humidity': 0.15,           # Saturación atmosférica
        'flow_velocity': 0.12,      # Velocidad del agua
        'sediments': 0.08,          # Obstrucción del cauce
        'pressure': 0.05,           # Condiciones atmosféricas
        'wind_speed': 0.03,         # Influencia menor
        'temperature': 0.02         # Influencia indirecta
    }
    
    # Calcular riesgo base
    risk = (
        norm_precipitation * weights['precipitation'] +
        norm_flow_volume * weights['flow_volume'] +
        norm_humidity * weights['humidity'] +
        norm_flow_velocity * weights['flow_velocity'] +
        norm_sediments * weights['sediments'] +
        norm_pressure * weights['pressure'] +
        norm_wind_speed * weights['wind_speed'] +
        temp_risk * weights['temperature']
    )
    
    # Factores de sinergia (condiciones que se potencian mutuamente)
    if precipitation > 100 and humidity > 90:
        risk *= 1.25  # Lluvia intensa con alta humedad
    
    if flow_volume > 1200 and flow_velocity > 5:
        risk *= 1.3  # Caudal alto con velocidad alta
    
    if sediments > 300 and flow_volume > 800:
        risk *= 1.15  # Obstrucción con alto caudal
    
    if pressure < 990 and precipitation > 80:
        risk *= 1.2  # Baja presión con lluvia fuerte
    
    # Limitar entre 0 y 100
    return min(max(risk * 100, 0), 100)

# =========================================================
# Generar datos históricos sintéticos
# =========================================================
def generate_historical_data(days=30):
    """Genera datos sintéticos para los últimos N días"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days*24, freq='H')
    
    data = {
        'fecha': dates,
        'precipitacion': np.random.gamma(2, 15, days*24),
        'temperatura': 15 + 10 * np.sin(np.arange(days*24) * 2 * np.pi / 24) + np.random.normal(0, 2, days*24),
        'humedad': np.clip(60 + np.random.normal(0, 15, days*24), 30, 100),
        'presion': 1013 + np.random.normal(0, 10, days*24),
        'viento': np.abs(np.random.normal(15, 10, days*24)),
        'caudal': 300 + 200 * np.sin(np.arange(days*24) * 2 * np.pi / (24*7)) + np.random.normal(0, 100, days*24),
    }
    
    df = pd.DataFrame(data)
    df['caudal'] = df['caudal'].clip(50, 2000)
    
    # Calcular riesgo para cada registro
    df['riesgo'] = df.apply(lambda row: calculate_flood_risk(
        row['presion'], row['humedad'], row['precipitacion'],
        2.0, row['viento'], row['temperatura'], 50, row['caudal']
    ), axis=1)
    
    return df

# =========================================================
# Título y descripción
# =========================================================
st.title("🌊 Simulador de Predicción de Inundaciones")
st.markdown('<p class="subtitle">Introduce valores o genera datos aleatorios para predecir el riesgo de inundación.</p>', unsafe_allow_html=True)

# =========================================================
# Sidebar con controles
# =========================================================
st.sidebar.header("⚙️ Parámetros de Simulación")

# Parámetros meteorológicos
st.sidebar.subheader("🌤️ Condiciones Meteorológicas")
pressure = st.sidebar.slider("Presión atmosférica (hPa)", 950, 1050, 1000, 1, 
                             help="Presión baja (<990) indica tormentas")
humidity = st.sidebar.slider("Humedad (%)", 0, 100, 83, 1,
                            help="Alta humedad favorece precipitación")
precipitation = st.sidebar.slider("Precipitación (mm)", 0, 200, 50, 5,
                                 help="Factor más crítico para inundaciones")
wind_speed = st.sidebar.slider("Velocidad del viento (km/h)", 0, 80, 18, 1,
                              help="Vientos fuertes intensifican tormentas")
temperature = st.sidebar.slider("Temperatura (°C)", 5, 40, 25, 1,
                               help="26-32°C óptimo para tormentas severas")

# Parámetros hidrológicos
st.sidebar.subheader("🌊 Condiciones Hidrológicas")
flow_velocity = st.sidebar.slider("Velocidad del caudal (m/s)", 0.0, 10.0, 2.0, 0.1,
                                  help="Mayor velocidad = mayor poder erosivo")
flow_volume = st.sidebar.slider("Volumen del caudal (m³/s)", 50, 2000, 504, 10,
                                help="Volumen de agua en el río")
sediments = st.sidebar.slider("Sedimentos (mg/L)", 0, 500, 50, 10,
                             help="Sedimentos pueden obstruir el cauce")

# Botones de presets
st.sidebar.subheader("🎯 Configuraciones Rápidas")
col1, col2 = st.sidebar.columns(2)

if col1.button("🟢 Riesgo Bajo", use_container_width=True):
    pressure = 1015
    humidity = 60
    precipitation = 10
    flow_velocity = 1.5
    wind_speed = 8
    temperature = 20
    sediments = 30
    flow_volume = 200

if col2.button("🔴 Riesgo Crítico", use_container_width=True):
    pressure = 985
    humidity = 95
    precipitation = 180
    flow_velocity = 7.5
    wind_speed = 55
    temperature = 29
    sediments = 400
    flow_volume = 1800

# =========================================================
# Calcular riesgo actual
# =========================================================
flood_risk = calculate_flood_risk(pressure, humidity, precipitation, flow_velocity,
                                  wind_speed, temperature, sediments, flow_volume)

# Determinar categoría
if flood_risk < 30:
    category = "BAJO"
    color = "green"
    emoji = "✅"
    message = "Las condiciones actuales no representan amenaza significativa."
elif flood_risk < 60:
    category = "MODERADO"
    color = "orange"
    emoji = "⚠️"
    message = "Se recomienda monitoreo constante de las condiciones."
elif flood_risk < 80:
    category = "ALTO"
    color = "red"
    emoji = "🔶"
    message = "Prepare medidas preventivas y alerte a la población."
else:
    category = "CRÍTICO"
    color = "darkred"
    emoji = "🚨"
    message = "Evacúe las zonas de riesgo inmediatamente. Situación crítica."

# =========================================================
# Visualización principal del riesgo
# =========================================================
st.markdown("---")
st.header("📊 Resultado de la Simulación")

# Medidor tipo gauge
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=flood_risk,
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': f"<b>Riesgo de Inundación</b><br><span style='font-size:0.8em'>Categoría: {category}</span>", 
           'font': {'size': 24}},
    number={'suffix': "%", 'font': {'size': 60}},
    gauge={
        'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "gray"},
        'bar': {'color': color, 'thickness': 0.75},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 30], 'color': '#d1fae5'},
            {'range': [30, 60], 'color': '#fed7aa'},
            {'range': [60, 80], 'color': '#fecaca'},
            {'range': [80, 100], 'color': '#fca5a5'}
        ],
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': flood_risk
        }
    }
))

fig_gauge.update_layout(
    height=400,
    margin=dict(l=20, r=20, t=80, b=20),
    paper_bgcolor="rgba(0,0,0,0)",
    font={'color': "#1f2937", 'family': "Arial"}
)

st.plotly_chart(fig_gauge, use_container_width=True)

# Mensaje de alerta
if flood_risk < 30:
    st.success(f"{emoji} **{category} RIESGO DE INUNDACIÓN** — {message}")
elif flood_risk < 60:
    st.warning(f"{emoji} **{category} RIESGO DE INUNDACIÓN** — {message}")
elif flood_risk < 80:
    st.error(f"{emoji} **{category} RIESGO DE INUNDACIÓN** — {message}")
else:
    st.error(f"{emoji} **{category} RIESGO DE INUNDACIÓN** — {message}")

# =========================================================
# Métricas en columnas
# =========================================================
st.markdown("---")
st.subheader("📈 Valores Actuales de Parámetros")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🌧️ Precipitación", f"{precipitation} mm", 
             delta=f"{precipitation - 50:+.0f} vs promedio")
    st.metric("🌡️ Temperatura", f"{temperature}°C",
             delta=f"{temperature - 25:+.0f} vs promedio")

with col2:
    st.metric("💧 Humedad", f"{humidity}%",
             delta=f"{humidity - 70:+.0f} vs promedio")
    st.metric("⛅ Presión", f"{pressure} hPa",
             delta=f"{pressure - 1013:+.0f} vs normal")

with col3:
    st.metric("🌊 Caudal", f"{flow_volume} m³/s",
             delta=f"{flow_volume - 500:+.0f} vs promedio")
    st.metric("⚡ Velocidad", f"{flow_velocity} m/s",
             delta=f"{flow_velocity - 2.0:+.1f} vs promedio")

with col4:
    st.metric("🌬️ Viento", f"{wind_speed} km/h",
             delta=f"{wind_speed - 15:+.0f} vs promedio")
    st.metric("🪨 Sedimentos", f"{sediments} mg/L",
             delta=f"{sediments - 100:+.0f} vs promedio")

# =========================================================
# Datos históricos
# =========================================================
st.markdown("---")
st.subheader("📅 Análisis Histórico (últimos 30 días)")

df_historical = generate_historical_data(30)

# Gráfico de riesgo histórico
fig_risk = px.area(df_historical, x='fecha', y='riesgo',
                   title='Evolución del Riesgo de Inundación',
                   labels={'fecha': 'Fecha', 'riesgo': 'Riesgo (%)'},
                   color_discrete_sequence=['#0ea5e9'])

fig_risk.add_hline(y=30, line_dash="dash", line_color="orange", 
                   annotation_text="Umbral Moderado")
fig_risk.add_hline(y=60, line_dash="dash", line_color="red",
                   annotation_text="Umbral Alto")

fig_risk.update_layout(height=400, hovermode='x unified')
st.plotly_chart(fig_risk, use_container_width=True)

# Gráficos de parámetros
col1, col2 = st.columns(2)

with col1:
    fig_precip = px.line(df_historical, x='fecha', y='precipitacion',
                        title='Precipitación Histórica',
                        labels={'fecha': 'Fecha', 'precipitacion': 'Precipitación (mm)'})
    fig_precip.update_traces(line_color='#3b82f6')
    fig_precip.update_layout(height=300)
    st.plotly_chart(fig_precip, use_container_width=True)

with col2:
    fig_caudal = px.line(df_historical, x='fecha', y='caudal',
                        title='Volumen del Caudal Histórico',
                        labels={'fecha': 'Fecha', 'caudal': 'Caudal (m³/s)'})
    fig_caudal.update_traces(line_color='#06b6d4')
    fig_caudal.update_layout(height=300)
    st.plotly_chart(fig_caudal, use_container_width=True)

# =========================================================
# Tabla de estadísticas
# =========================================================
st.subheader("📊 Estadísticas Resumidas")

stats_data = {
    'Variable': ['Precipitación', 'Temperatura', 'Humedad', 'Presión', 'Caudal', 'Riesgo'],
    'Mínimo': [
        f"{df_historical['precipitacion'].min():.1f} mm",
        f"{df_historical['temperatura'].min():.1f}°C",
        f"{df_historical['humedad'].min():.1f}%",
        f"{df_historical['presion'].min():.1f} hPa",
        f"{df_historical['caudal'].min():.1f} m³/s",
        f"{df_historical['riesgo'].min():.1f}%"
    ],
    'Promedio': [
        f"{df_historical['precipitacion'].mean():.1f} mm",
        f"{df_historical['temperatura'].mean():.1f}°C",
        f"{df_historical['humedad'].mean():.1f}%",
        f"{df_historical['presion'].mean():.1f} hPa",
        f"{df_historical['caudal'].mean():.1f} m³/s",
        f"{df_historical['riesgo'].mean():.1f}%"
    ],
    'Máximo': [
        f"{df_historical['precipitacion'].max():.1f} mm",
        f"{df_historical['temperatura'].max():.1f}°C",
        f"{df_historical['humedad'].max():.1f}%",
        f"{df_historical['presion'].max():.1f} hPa",
        f"{df_historical['caudal'].max():.1f} m³/s",
        f"{df_historical['riesgo'].max():.1f}%"
    ]
}

st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

# =========================================================
# Footer
# =========================================================
st.markdown("---")
st.caption("🔬 Modelo de predicción basado en algoritmos hidrometeorológicos | 📊 Datos sintéticos generados internamente")
