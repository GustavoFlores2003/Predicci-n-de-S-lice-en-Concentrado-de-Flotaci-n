import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# Configuración de la página del dashboard
st.set_page_config(
    page_title="Monitoreo y Simulación — Planta de Flotación Minera",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar el modelo guardado
@st.cache_resource
def load_best_model():
    # Ruta relativa al archivo actual
    model_path = os.path.join(os.path.dirname(__file__), '../models/best_model.joblib')
    if not os.path.exists(model_path):
        model_path = 'models/best_model.joblib'
    return joblib.load(model_path)

# Cargar el conjunto de datos de prueba
@st.cache_data
def load_test_data():
    data_path = os.path.join(os.path.dirname(__file__), '../data/processed/test_engineered.parquet')
    if not os.path.exists(data_path):
        data_path = 'data/processed/test_engineered.parquet'
    return pd.read_parquet(data_path)

# Cargar recursos
try:
    model_dict = load_best_model()
    model = model_dict['model']
    features = model_dict['features']
    has_model = True
except Exception as e:
    has_model = False
    model_error = str(e)

try:
    test_df = load_test_data()
    has_data = True
except Exception as e:
    has_data = False
    data_error = str(e)

# Título y Encabezado Principal
st.title("⛏️ Control de Calidad del Concentrado — Proceso de Flotación")
st.markdown(
    """
    Este dashboard permite simular el porcentaje final de sílice en el concentrado (**% Silica Concentrate**) a partir de variables de alimentación del proceso y variables operativas controlables.
    Además, permite monitorear el desempeño del modelo en el periodo histórico de validación.
    """
)
st.markdown("---")

if not has_model:
    st.error(f"Error al cargar el modelo best_model.joblib en 'models/': {model_error}")
    st.stop()

# --- BARRA LATERAL (SIMULADOR DE PARÁMETROS) ---
st.sidebar.title("🎛️ Panel de Simulación (What-If)")
st.sidebar.markdown("Ajusta las variables operativas en tiempo real para simular el punto de operación estable de la planta.")

# Umbral de calidad
threshold = st.sidebar.slider("🚨 Umbral de Alerta de Sílice (%)", 1.0, 5.0, 3.0, 0.1)

# Sliders agrupados en expanders
with st.sidebar.expander("Feed (Alimentación de Mineral)"):
    iron_feed = st.slider("% Hierro en Alimentación (% Iron Feed)", 42.7, 65.8, 53.5, 0.1)
    silica_feed = st.slider("% Sílice en Alimentación (% Silica Feed)", 1.3, 33.4, 18.0, 0.1)
    pulp_flow = st.slider("Caudal de Pulpa (Ore Pulp Flow) (m³/h)", 376.0, 419.0, 388.8, 0.5)
    pulp_density = st.slider("Densidad de Pulpa (Ore Pulp Density) (g/cm³)", 1.50, 1.86, 1.69, 0.01)

with st.sidebar.expander("Reactivos y pH (Variables de Control)"):
    starch_flow = st.slider("Flujo de Almidón (Starch Flow) (g/h)", 0.0, 6300.0, 3059.0, 10.0)
    amina_flow = st.slider("Flujo de Amina (Amina Flow) (g/h)", 241.0, 740.0, 481.0, 1.0)
    pulp_ph = st.slider("pH de la Pulpa (Ore Pulp pH)", 8.7, 10.7, 9.6, 0.1)

with st.sidebar.expander("Flujos de Aire en Columnas (Variables de Control)"):
    air_flows = []
    for i in range(1, 8):
        val = st.slider(f"Aire Columna {i} (Air Flow)", 175.0, 376.0, 300.0, 1.0, key=f"air_{i}")
        air_flows.append(val)

with st.sidebar.expander("Niveles de Espuma en Columnas (Variables de Control)"):
    levels = []
    for i in range(1, 8):
        val = st.slider(f"Nivel Columna {i} (Level)", 120.0, 887.0, 500.0, 1.0, key=f"level_{i}")
        levels.append(val)


# --- CREACIÓN DEL VECTOR DE CARACTERÍSTICAS PARA SIMULACIÓN ---
# Crear diccionario con los valores crudos configurados en los sliders
raw_simulated = {
    '% Iron Feed': iron_feed,
    '% Silica Feed': silica_feed,
    'Starch Flow': starch_flow,
    'Amina Flow': amina_flow,
    'Ore Pulp Flow': pulp_flow,
    'Ore Pulp pH': pulp_ph,
    'Ore Pulp Density': pulp_density
}
for i in range(1, 8):
    raw_simulated[f'Flotation Column {i:02d} Air Flow'] = air_flows[i-1]
    raw_simulated[f'Flotation Column {i:02d} Level'] = levels[i-1]

# Reconstruir las 117 variables que el modelo espera (incluyendo promedios y desviaciones móviles)
# Asumimos que los promedios móviles históricos se encuentran en estado estacionario (iguales al punto actual)
# y que las desviaciones estándar de las ventanas móviles son 0.0 (sin variabilidad histórica)
simulated_row = {}
for feat in features:
    if feat in raw_simulated:
        simulated_row[feat] = raw_simulated[feat]
    elif '_roll_mean_' in feat:
        base_col = feat.split('_roll_mean_')[0]
        simulated_row[feat] = raw_simulated[base_col]
    elif '_roll_std_' in feat:
        simulated_row[feat] = 0.0
    else:
        simulated_row[feat] = 0.0

# DataFrame para la inferencia
simulated_df = pd.DataFrame([simulated_row])[features]


# --- SECCIÓN PRINCIPAL: PESTAÑAS ---
tab1, tab2, tab3 = st.tabs([
    "⚡ Simulación en Tiempo Real", 
    "📈 Serie Histórica (Real vs. Predicho)", 
    "ℹ️ Información del Proceso"
])

# Pestaña 1: Simulación en Tiempo Real
with tab1:
    st.subheader("⚡ Análisis de Inferencia en Tiempo Real")
    
    # Realizar predicción con el modelo
    pred_silica = model.predict(simulated_df)[0]
    
    # Organizar layouts en columnas
    col_card, col_gauge = st.columns([1, 1])
    
    with col_card:
        st.markdown("<br>", unsafe_allow_html=True)
        # Mostrar el valor en formato grande de métrica
        st.metric(
            label="Predicción de % de Sílice en Concentrado", 
            value=f"{pred_silica:.3f} %",
            delta=f"{(pred_silica - threshold):.3f} % vs Umbral" if pred_silica > threshold else f"{(pred_silica - threshold):.3f} % vs Umbral",
            delta_color="inverse"
        )
        
        # Alerta visual (Semáforo de calidad)
        if pred_silica > threshold:
            st.markdown(
                f"""
                <div style="background-color: #ffe6e6; border: 3px solid #ff4d4d; border-radius: 12px; padding: 20px; text-align: center; margin-top: 15px;">
                    <span style="font-size: 60px; color: #ff4d4d; animation: blinker 1.5s linear infinite;">🔴</span>
                    <h3 style="color: #990000; margin-top: 10px; font-weight: bold;">ALERTA: Sílice Fuera de Rango</h3>
                    <p style="color: #660000; font-size: 15px; margin-bottom: 0px;">
                        La predicción actual de <b>{pred_silica:.3f}%</b> excede el umbral de especificación comercial de <b>{threshold:.2f}%</b>. 
                        Aumentar el colector (Amina) o ajustar los flujos de aire puede ayudar a deprimir más sílice.
                    </p>
                </div>
                <style>
                @keyframes blinker {{
                    50% {{ opacity: 0; }}
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="background-color: #e6ffe6; border: 3px solid #2eb82e; border-radius: 12px; padding: 20px; text-align: center; margin-top: 15px;">
                    <span style="font-size: 60px; color: #2eb82e;">🟢</span>
                    <h3 style="color: #1f7a1f; margin-top: 10px; font-weight: bold;">Calidad de Concentrado Conforme</h3>
                    <p style="color: #145214; font-size: 15px; margin-bottom: 0px;">
                        La calidad del producto cumple con las especificaciones comerciales de pureza (Predicción: <b>{pred_silica:.3f}%</b> &le; Umbral: <b>{threshold:.2f}%</b>).
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    with col_gauge:
        # Gráfico de termómetro/indicador analógico de calidad
        fig_indicator = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = pred_silica,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "% Sílice Predicho", 'font': {'size': 20}},
            gauge = {
                'axis': {'range': [0, 6], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, threshold], 'color': 'rgba(100, 230, 100, 0.4)'},
                    {'range': [threshold, 6], 'color': 'rgba(255, 100, 100, 0.4)'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold
                }
            }
        ))
        fig_indicator.update_layout(height=350, margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig_indicator, use_container_width=True)

    # Mostrar influencia rápida del punto de operación
    st.markdown("### 📊 Operación Actual Evaluada")
    st.write("Las características simuladas representan un estado estable constante. En la barra lateral puedes modificar de forma detallada cada una de ellas.")


# Pestaña 2: Serie Histórica (Real vs. Predicho)
with tab2:
    st.subheader("📈 Validación Histórica del Modelo en Producción (Test Cronológico)")
    
    if not has_data:
        st.error(f"Error al cargar el archivo data/processed/test_engineered.parquet: {data_error}")
    else:
        st.markdown(
            """
            Esta pestaña evalúa las predicciones históricas generadas en el set de prueba cronológico (no aleatorio). 
            Los datos representan las últimas semanas de operación y son utilizados para validar la capacidad del modelo ante derivas temporales reales.
            """
        )
        
        # Generar predicciones para todo el conjunto de prueba
        with st.spinner("Generando inferencia para todo el set de pruebas..."):
            X_test = test_df[features].copy()
            y_test = test_df['% Silica Concentrate'].copy()
            y_pred = model.predict(X_test)
            
            test_results = test_df[['date', '% Silica Concentrate']].copy()
            test_results['Predicción % Silica Concentrate'] = y_pred
            test_results = test_results.rename(columns={'% Silica Concentrate': 'Real % Silica Concentrate'})
        
        # Promediar a nivel diario para visualización óptima e interactividad sin lag
        test_daily = test_results.set_index('date').resample('D').mean().reset_index()
        
        # Gráfico interactivo con Plotly
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=test_daily['date'], 
            y=test_daily['Real % Silica Concentrate'],
            mode='lines+markers', 
            name='Real (Media Diaria)', 
            line=dict(color='#1f77b4', width=2.5)
        ))
        fig_ts.add_trace(go.Scatter(
            x=test_daily['date'], 
            y=test_daily['Predicción % Silica Concentrate'],
            mode='lines+markers', 
            name='Predicción (Media Diaria)', 
            line=dict(color='#ff7f0e', width=2.5, dash='dash')
        ))
        
        fig_ts.update_layout(
            title="Evolución Temporal de Calidad: Valor Real vs Predicción del Modelo",
            xaxis_title="Fecha de Operación",
            yaxis_title="% Sílice en el Concentrado",
            hovermode="x unified",
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)'),
            height=450,
            margin=dict(t=50, b=50, l=10, r=10)
        )
        
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # Métricas principales calculadas a nivel individual
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Tarjetas de métricas
        st.markdown("#### 🎯 Métricas de Evaluación Globales (Nivel de Observación)")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("MAE (Error Absoluto Medio)", f"{mae:.4f}")
        m_col2.metric("RMSE (Error Cuadrático Medio)", f"{rmse:.4f}")
        m_col3.metric("R² (Coeficiente de Determinación)", f"{r2:.4f}")
        
        st.info(
            """
            💡 **Nota de Metalurgia & Data Science:** El R² es ligeramente negativo debido a la estricta validación cronológica y el alto nivel de ruido del proceso a nivel horario.
            Esto demuestra que el modelo enfrenta una deriva temporal y que el sistema requiere adaptabilidad, sirviendo como indicador honesto para operadores comerciales en lugar de un split aleatorio sesgado.
            """
        )

# Pestaña 3: Información del Proceso
with tab3:
    st.subheader("ℹ️ Contexto Metalúrgico y Variables del Proceso")
    st.markdown(
        """
        El proceso de **flotación inversa** de mineral de hierro consiste en atrapar las impurezas de sílice (cuarzo) en burbujas de espuma para hacerlas flotar y retirarlas, permitiendo que el hierro concentrado y puro decante al fondo de las columnas.
        
        ### Variables de Proceso Clave:
        
        * **% Silica Feed / % Iron Feed:** Representan las leyes iniciales del mineral de hierro que alimenta a la planta. Son variables perturbadoras físicas que no pueden ser controladas de forma directa.
        * **Amina (Colector):** Reactivo químico dosificado para adherirse selectivamente a las partículas de sílice y hacerlas hidrofóbicas para que floten. A mayor flujo de amina (`Amina Flow`), menor sílice final (mayor pureza), pero a un costo operativo mayor.
        * **Almidón (Depresor):** Reactivo químico dosificado para inhibir la flotación de las partículas de hierro, obligándolas a decantar al fondo.
        * **pH de la Pulpa:** Mantenido en rangos específicos (usualmente básicos, pH ~9.5) para maximizar la selectividad de la amina sobre el cuarzo.
        * **Flujo de Aire (Column Air Flow):** Controla el volumen y tamaño de las burbujas en las columnas de flotación. Flujos excesivos arrastran hierro valioso, flujos deficientes no logran remover suficiente sílice.
        * **Nivel de Espuma (Column Level):** La altura de la interfaz pulpa-espuma en la columna. Controla el tiempo de residencia de la espuma antes del rebose.
        """
    )
