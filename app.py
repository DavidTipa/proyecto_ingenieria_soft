# Librerías básicas
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Visualización
import seaborn as sns
import plotly.express as px

# Modelado estadístico
import statsmodels.api as sm

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Estadísticas (opcional)
import scipy.stats as stats

# Configuración de página
st.set_page_config(
    page_title="Dashboard Analítico - Oslo Airbnb",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS
st.markdown("""
    <style>
      
.main {
    background-color: rgba(255, 255, 255, 0.85); /* fondo blanco semi-transparente dentro */
    padding: 20px;
    border-radius: 15px;
}
    .sidebar .sidebar-content {
        background-color: #2C3E50; 
        color: white;
    }
    .stSelectbox, .stMultiSelect {margin-bottom: 15px;}
    .plot-container {
        background-color: white; 
        padding: 15px; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    [data-testid="stExpander"] .st-emotion-cache-1ck1owl {
        background-color: #FFFFFF;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .intro-header {
        font-size: 2.5em;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 30px;
    }
    .intro-card {
        background-color: white;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .start-button {
        display: block;
        width: 200px;
        margin: 40px auto;
        padding: 15px;
        font-size: 1.2em;
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        text-align: center;
    }
    .start-button:hover {
        background-color: #2980b9;
    }
    .data-preview {
        font-size: 0.8em;
        margin-top: 10px;
    }
    .data-preview table {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Estado de sesión para controlar si mostramos la introducción o el dashboard
if 'show_dashboard' not in st.session_state:
    st.session_state.show_dashboard = False

def show_intro():
    # CSS para el fondo con imagen y overlay
    st.markdown("""
     <style>
                .stApp {
        background-image: url("https://wallpapers.com/images/high/viking-pictures-of5u0x5ii0qcgxhe.webp");
        background-size: cover;
        background-attachment: fixed;  /* Efecto parallax */
        background-position: center;
    }
    /* CONTENEDOR PRINCIPAL CON IMAGEN DE FONDO */
    .intro-container {
        
        background-size: cover;
        background-position: left;
        padding: 40px;
        border-radius: 10px;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;  /* Importante para que las líneas no se salgan */
    }

    /* OVERLAY ROJO (TRANSPARENTE PARA VER EL FONDO) */
    .intro-overlay {
        background-color: rgba(209, 19, 47, 0.7);  /* Rojo con 70% de opacidad */
        padding: 40px;
        border-radius: 10px;
        position: relative;
        
    }

    /* TUS LÍNEAS DECORATIVAS (EXACTAMENTE COMO LAS TIENES) */
    .intro-overlay::before {
        content: "";
        position: absolute;
        left: 20%;
        top: 0%;
        bottom: 24%;
        width: 17px;
        background: #00205B;
        border-right: 4px solid white;
        border-left: 4px solid white;
        box-sizing: border-box;
    }
    
    .intro-overlay::after {
        content: "";
        position: absolute;
        left: 20%;
        top: 72.7%;
        bottom: 0%;
        width: 17px;
        background: #00205B;
        border-right: 4px solid white;
        border-left: 4px solid white;
        box-sizing: border-box;
    }
    
    .horizontal-line {
        position: absolute;
        left: 0%;
        right: 0%;
        top: 65%;
        height: 17px;
        background: #00205B;
        border: 4px solid white;
        box-sizing: border-box;
    }
        .intro-button-container {
        text-align: center;
        margin-top: 30px;
        
    }
    .intro-title {
        color: white;
        text-align: center;
        margin-bottom: 30px;
        font-size: 2.5em;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
    }
    .intro-button {
        background: linear-gradient(45deg, #00205B, #D1132F);
        color: white !important;
        font-weight: bold;
        border: none;
        padding: 15px 30px;
        font-size: 18px;
        border-radius: 50px;
        cursor: pointer;
        transition: all 0.3s;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        display: block;
        margin: 30px auto 0;
        width: fit-content;
    }
    .intro-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
     .info-card {
        
        padding: 25px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        height: 100%;
    }
    .info-card h3 { 
        color: white;
        border-bottom: 2px solid #D1132F;
        padding-bottom: 10px;
    }
    .info-list {
        columns: 2;
        list-style-type: none;
        padding-left: 0;
        font-size: 1.05em;
    }
    .info-list li {
        margin-bottom: 10px;
        display: flex;
        align-items: center;
    }
    .info-list li:before {
        content: "•";
        color: #D1132F;
        font-weight: bold;
        display: inline-block;
        width: 1em;
        margin-left: -1em;
    }

    /* FOOTER */
    .intro-footer {
        text-align: center;
        margin-top: 30px;
        color: white;
        font-size: 0.9em;
        opacity: 0.8;
    }
    
    </style>
    """, unsafe_allow_html=True)

    # Coordenadas de Oslo (latitud, longitud)
oslo_coords = [59.9139, 10.7522]

# Crear dataframe con la ubicación
map_data = pd.DataFrame({
    'lat': [oslo_coords[0]],
    'lon': [oslo_coords[1]]
})

# Inicializar estado de la aplicación
if 'show_dashboard' not in st.session_state:
    st.session_state.show_dashboard = False

# Mostrar presentación si aún no se ha iniciado el análisis
if not st.session_state.show_dashboard:
    st.markdown("""
    <div class="intro-container">
        <div class="intro-overlay">
            <h1 class="intro-title">Análisis de datos Airbnb, Oslo</h1>
            <div class="horizontal-line"></div>
    """, unsafe_allow_html=True)

    cols = st.columns([3, 3, 2, 2, 2])

    with cols[0]:
        with st.container():
            st.markdown('<div class="plotly-container">', unsafe_allow_html=True)
            fig = px.scatter_map(
                map_data,
                lat="lat",
                lon="lon",
                zoom=11,
                color_discrete_sequence=["#D1132F"],
                size=[15]
            )
            fig.update_mapboxes(
                style="open-street-map",
                center=dict(lat=oslo_coords[0], lon=oslo_coords[1])
            )
            fig.update_layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("📍 Mapa de Oslo, Noruega - Capital cultural y económica")
            st.markdown('</div>', unsafe_allow_html=True)

    with cols[1]:
        st.markdown("""
        <div class="info-card">
            <h3 >Oslo en el mundo</h3>
            <p style="font-size: 1.05em; color: #FFFFFF;">Capital de Noruega que combina naturaleza, cultura y desarrollo:</p>
            <ul class="info-list">
                <li>🏛️ Centro financiero y naval</li>
                <li>🌿 Ciudad más sostenible</li>
                <li>🎭 50+ museos y galerías</li>
                <li>🏆 Mejor calidad de vida</li>
                <li>🛳️ Arquitectura premiada</li>
                <li>🍽️ 15 restaurantes Michelin</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        if st.button('COMENZAR ANÁLISIS', key='start_analysis'):
            st.session_state.show_dashboard = True
            st.rerun()

    with cols[2]:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            fig_pie = px.pie(
                names=["Apartamentos", "Casas", "Otros"],
                values=[65, 25, 10],
                title="<b>Tipos de alojamiento</b>",
                hole=0.5,
                color_discrete_sequence=["#D1132F", "#FF6B6B", "#FFA8A8"]
            )
            fig_pie.update_layout(
                showlegend=True,
                height=250,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3),
                margin={"t": 40, "b": 20},
                title_x=0.5
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with cols[3]:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            fig_bar = px.bar(
                x=["Centro", "Grünerløkka", "Frogner"],
                y=[1200, 950, 1100],
                title="<b>Precio promedio (NOK)</b>",
                color=["#D1132F", "#FF6B6B", "#FF6B6B"],
                labels={'x': '', 'y': ''}
            )
            fig_bar.update_layout(
                height=250,
                plot_bgcolor='rgba(0,0,0,0)',
                margin={"t": 40, "b": 20},
                title_x=0.5,
                xaxis_tickangle=-30
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.caption("💵 Precios por noche en temporada alta")
            st.markdown('</div>', unsafe_allow_html=True)

    with cols[4]:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 🌡️ Clima y Demografía")
            st.metric("Temperatura media", "5.7°C", "1.2°C vs 2022")
            st.markdown("---")
            st.markdown("### 🏙️ Población")
            st.metric("Habitantes", "697,010", "2.1% crecimiento")
            st.markdown("---")
            st.markdown("### 🏨 Airbnb")
            st.metric("Propiedades", "4,850", "12% ▲")
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

# Aquí puedes agregar el contenido que debe mostrarse después de hacer clic en "COMENZAR ANÁLISIS"
if st.session_state.show_dashboard:
     
    

# Función para gráficos de regresión
 def create_regression_plot(x, y, predictions, x_label, y_label, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x, y, alpha=0.6, color='#3498db', label='Datos reales')
    
    if len(x.shape) == 1:
        sorted_idx = np.argsort(x)
        ax.plot(x[sorted_idx], predictions[sorted_idx], 
                color='#e74c3c', linewidth=2, label='Línea de regresión')
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    return fig

# Carga de datos
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('listings_50_outliers_oslo2.csv')
        if 'room_type' in df.columns:
            df['room_type_code'] = pd.factorize(df['room_type'])[0]
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {str(e)}")
        return pd.DataFrame()

# Función principal del dashboard
def main_dashboard():
    df = load_data()

    # Sidebar - Configuración
    with st.sidebar:
        st.title("Seleccion de Análisis")
        st.image("https://via.placeholder.com/150x50?text=Analytics", use_container_width=True)
        
        # Vista previa de datos en el sidebar
        if not df.empty:
            with st.expander("🔍 Vista previa de datos (5 registros)", expanded=True):
                st.dataframe(df.head().style.set_properties(**{
                    'font-size': '0.8em',
                    'padding': '2px'
                }))
        
        analysis_section = st.radio(
            "Sección de Análisis",
            options=["Análisis de Regresión", "Análisis Variado"]
        )
        
        if analysis_section == "Análisis de Regresión":
            analysis_type = st.selectbox(
                "Tipo de análisis",
                options=["Regresión Lineal", "Regresión Múltiple", "Regresión Logística"]
            )
            
            if not df.empty:
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                
                if analysis_type == "Regresión Lineal":
                    st.subheader("Variables para Regresión Lineal")
                    x_var = st.selectbox("Variable independiente (X)", options=numeric_cols)
                    y_var = st.selectbox("Variable dependiente (Y)", options=numeric_cols)
                    
                elif analysis_type == "Regresión Múltiple":
                    st.subheader("Variables para Regresión Múltiple")
                    y_var = st.selectbox("Variable dependiente (Y)", options=numeric_cols)
                    x_vars = st.multiselect("Variables independientes (X)", options=numeric_cols)
                    
                elif analysis_type == "Regresión Logística":
                    st.subheader("Variables para Regresión Logística")
                    columnas_categoricas_permitidas = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified','instant_bookable']
                    columnas_numericas_permitidas = [ 'bathrooms', 'bedrooms', 'beds', 'price','minimum_nights', 'maximum_nights']

                    columnas_categoricas = [col for col in columnas_categoricas_permitidas if col in df.columns]
                    if not columnas_categoricas:
                        st.warning("No se encontraron variables categóricas en el dataset")
                    else:
                        y_var = st.selectbox("Variable dependiente (Y)", options=columnas_categoricas)
                        x_vars = st.multiselect("Variables independientes (X)", columnas_numericas_permitidas)
                    
        
        st.markdown("---")
        st.markdown("🔍 **Instrucciones:**")
        st.markdown("1. Selecciona la sección de análisis")
        st.markdown("2. Elige las variables correspondientes")
        st.markdown("3. Explora los resultados visuales")

    # Contenido principal
    st.title("📈 Dashboard Analítico - Airbnb Oslo")
    

    if df.empty:
        st.warning("No se han cargado datos. Verifica tu archivo CSV.")
    else:
        if analysis_section == "Análisis de Regresión":
            st.subheader(f"📊 Resultados: {analysis_type}")
            
            try:
                if analysis_type == "Regresión Lineal":
                    if 'x_var' in locals() and 'y_var' in locals():
                        X = sm.add_constant(df[x_var])
                        y = df[y_var]
                        
                        model = sm.OLS(y, X).fit()
                        
                        with st.container():
                            st.markdown("### 📈 Gráfico de Regresión")
                            fig = create_regression_plot(
                                df[x_var].values, 
                                y.values,
                                model.predict(X),
                                x_var,
                                y_var,
                                f"Regresión Lineal: {y_var} ~ {x_var}"
                            )
                            st.pyplot(fig)
                        
                elif analysis_type == "Regresión Múltiple":
                    if 'x_vars' in locals() and len(x_vars) > 0 and 'y_var' in locals():
                        
                        df_clean=df[x_vars+[y_var]].dropna()
                        X= df_clean[x_vars]
                        y= df_clean[y_var]


                        
                        X= sm.add_constant(X)
                        model = sm.OLS(y, X).fit()
                        
                        
                        with st.container():
                             st.markdown("### 📝 Resumen del Modelo")
                             col1, col2, col3 = st.columns(3)
                             with col1:
                                st.metric("R²", f"{model.rsquared:.3f}")
                             with col2:
                                st.metric("R² Ajustado", f"{model.rsquared_adj:.3f}")
                             with col3:
                                st.metric("Variables", f"{len(x_vars)}")
                        with st.container():
                            st.markdown("### 📊 Relaciones Individuales con la Variable Dependiente")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            colors= ["#f6a0a0", "orange", "#ebc9a4", "brown", "#a0c4f6", "#a0f6b2"]
                            for i, var in enumerate(x_vars):
                                sns.scatterplot(
                                    x=var,
                                    y=y_var,
                                    data=df_clean,
                                    color=colors[i % len(colors)],
                                    alpha=0.6,
                                    ax=ax,
                                    label=f"{var} vs {y_var}"
                    

                                )
                            for var in x_vars:
                                sns.regplot(
                                    x=var,
                                    y=y_var,
                                    data=df_clean,
                                    scatter=False,
                                    color='black',
                                    ax=ax,
                                    line_kws={'linestyle': '--', 'alpha': 0.3},
                                )
                            ax.set_title(f"Relaciones: {y_var} vs Variables Independientes")
                            ax.legend()
                            ax.grid(True, linestyle='--', alpha=0.7)
                            st.pyplot(fig)
                
            
                        with st.container():
                             st.markdown("### 📊 Coeficientes del Modelo")
                             coef_df = pd.DataFrame({
                                'Variable': ['Intercepto'] + x_vars,
                                'Coeficiente': model.params,
                                'P-valor': model.pvalues
                            }).set_index('Variable')
                        
                        with st.container():
                            st.markdown("### matriz de correlación")
                            corr_matrix = df_clean.corr()
                            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu', zmin=-1, zmax=1,title="Matriz de Correlación")
                            st.plotly_chart(fig, use_container_width=True)
                           
                            
                       
                        
                       
                        
                elif analysis_type == "Regresión Logística":
                    if 'x_vars' in locals() and len(x_vars) > 0 and 'y_var' in locals():
                        if len(df[y_var].unique()) != 2:
                            st.error("La variable objetivo debe ser binaria (2 valores únicos)")
                        else:
                            df_clean = df[x_vars + [y_var]].dropna()
                            X = df_clean[x_vars]
                            y = df_clean[y_var]


                           
            
                            from sklearn.model_selection import train_test_split
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


                            model = LogisticRegression(
                                  max_iter=1000,
                                  class_weight='balanced',  # Manejo de clases desbalanceadas
                                   random_state=42
                                   ).fit(X_train, y_train)         
                                    
                                   
            
        
            # Predecir en datos de prueba
                            y_pred = model.predict(X_test)
            
            # --- MATRIZ DE CONFUSIÓN ---
                            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
                            st.markdown("### 📊 Matriz de Confusión")
                            cm = confusion_matrix(y_test, y_pred)
                            fig, ax = plt.subplots(figsize=(4, 4))
                            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=model.classes_)
                            disp.plot(cmap="Blues", ax=ax, colorbar=False)
                          
                            st.pyplot(fig)
                            
            
            
                    
            except Exception as e:
                st.error(f"Error en el análisis: {str(e)}")
        
        else:  # Análisis Categórico
            st.subheader("📊 Análisis General de variables")
            
            categorical_cols = df.select_dtypes(include=['object', 'category','float64']).columns.tolist()
            
            if not categorical_cols:
                st.warning("No se encontraron variables categóricas en el dataset")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    selected_cat_var = st.selectbox(
                        "Selecciona variables",
                        options=categorical_cols
                    )
                with col2:
                    chart_type = st.selectbox(
                        "Tipo de gráfico",
                        options=["Bar Plot", "Pie Plot", "Box Plot", "Line Plot"]
                    )
                
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                selected_num_var = st.selectbox(
                    "Selecciona variable numérica (opcional)",
                    options=["Ninguna"] + numeric_cols
                )
                
                # Gráficos principales
                with st.container():
                    st.markdown(f"### 📈 Visualización: {selected_cat_var}")
                    
                    if chart_type == "Bar Plot":
                        if selected_num_var == "Ninguna":
                            plot_data = df[selected_cat_var].value_counts().reset_index()
                            plot_data.columns = [selected_cat_var, 'count']

                            fig = px.bar(
                                plot_data,
                                x=selected_cat_var,
                                y='count',
                                title=f"Conteo por {selected_cat_var}",
                                labels={'count': 'Conteo'},
                                color=selected_cat_var,
                                text_auto=True
                            )
                        else:
                            plot_data = df.groupby(selected_cat_var)[selected_num_var].mean().reset_index()
                            fig = px.bar(
                                plot_data,
                                x=selected_cat_var,
                                y=selected_num_var,
                                title=f"Media de {selected_num_var} por {selected_cat_var}",
                                color=selected_cat_var,
                                text_auto='.2f'
                            )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif chart_type == "Pie Plot":
                        if selected_num_var == "Ninguna":
                            fig = px.pie(
                                df,
                                names=selected_cat_var,
                                title=f"Distribución de {selected_cat_var}",
                                hole=0.3
                            )
                        else:
                            fig = px.pie(
                                df.groupby(selected_cat_var)[selected_num_var].sum().reset_index(),
                                names=selected_cat_var,
                                values=selected_num_var,
                                title=f"Distribución de {selected_num_var} por {selected_cat_var}"
                            )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif chart_type == "Box Plot":
                        if selected_num_var != "Ninguna":
                            fig = px.box(
                                df,
                                x=selected_cat_var,
                                y=selected_num_var,
                                title=f"Distribución de {selected_num_var} por {selected_cat_var}",
                                color=selected_cat_var
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Se requiere seleccionar una variable numérica para Box Plot")
                    
                    elif chart_type == "Line Plot":
                        if selected_num_var != "Ninguna":
                            plot_data = df.groupby(selected_cat_var)[selected_num_var].mean().reset_index()
                            fig = px.line(
                                plot_data,
                                x=selected_cat_var,
                                y=selected_num_var,
                                title=f"Tendencia de {selected_num_var} por {selected_cat_var}",
                                markers=True
                            )
                            fig.update_traces(line=dict(width=2.5))
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Se requiere seleccionar una variable numérica para Line Plot")
                
    # Footer
    st.markdown("---")
    st.markdown("**Dashboard Analítico** | © 2025 - Jose David Chavez Tipa")

# Control de qué mostrar
if not st.session_state.show_dashboard:
    show_intro()
else:
    main_dashboard()