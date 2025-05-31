"""
Aplicación Streamlit para el Generador de Reseñas
Interface principal del usuario - VERSIÓN SIMPLIFICADA
"""

import streamlit as st
import config
from utils.text_generator import generate_movie_review, initialize_generator

# Configuración de la página
st.set_page_config(
    page_title="🎬 Generador de Reseñas",
    page_icon="🎬",
    layout="wide"
)

def main():
    """
    Función principal de la aplicación
    """
    # Título principal
    st.title("🎬 Generador de Reseñas de Películas")
    st.markdown("*Powered by Transformers - Genera reseñas automáticamente*")
    
    # Verificar que los modelos estén disponibles
    if 'models_initialized' not in st.session_state:
        with st.spinner("🤖 Cargando modelos entrenados..."):
            success = initialize_generator()
            st.session_state.models_initialized = success
            if success:
                st.success("✅ Modelos cargados exitosamente")
            else:
                st.error("❌ Error cargando modelos. Verifica que el entrenamiento se haya completado.")
                st.info("💡 Ejecuta: `python solucion_definitiva.py` para entrenar los modelos")
                return
    
    # Sidebar para configuraciones
    st.sidebar.header("⚙️ Configuraciones")
    
    # Selección de modelo
    modelo_seleccionado = st.sidebar.selectbox(
        "Selecciona el modelo:",
        ["model1", "model2"],
        format_func=lambda x: "Transformer Modelo 1" if x == "model1" else "Transformer Modelo 2",
        help="Modelo 1: Arquitectura básica, Modelo 2: Dos capas transformer"
    )
    
    # Parámetros de generación
    temperatura = st.sidebar.slider(
        "Temperatura (Creatividad):",
        min_value=0.1,
        max_value=2.0,
        value=config.DEFAULT_TEMPERATURE,
        step=0.1,
        help="Valores bajos = más conservador, valores altos = más creativo"
    )
    
    longitud = st.sidebar.slider(
        "Longitud de la reseña:",
        min_value=20,
        max_value=200,
        value=config.DEFAULT_LENGTH,
        step=10,
        help="Número aproximado de palabras a generar"
    )
    
    # Área principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("✍️ Escribe el inicio de tu reseña")
        
        # Input del usuario
        texto_inicial = st.text_area(
            "Inicio de la reseña:",
            value=st.session_state.get('texto_inicial', config.DEFAULT_PROMPT),
            height=100,
            help="Escribe las primeras palabras y el modelo completará la reseña"
        )
        
        # Botón para generar
        if st.button("🚀 Generar Reseña", type="primary"):
            if texto_inicial.strip():
                with st.spinner("🎬 Generando reseña con IA..."):
                    # Generar reseña
                    reseña_generada = generate_movie_review(
                        prompt=texto_inicial,
                        model_choice=modelo_seleccionado,
                        temperature=temperatura,
                        length=longitud
                    )
                    
                    # Mostrar resultado
                    if reseña_generada and not reseña_generada.startswith("❌"):
                        st.success("¡Reseña generada exitosamente!")
                        
                        # Mostrar la reseña generada
                        st.markdown("### 📝 Reseña Generada:")
                        st.markdown(f"""
                        <div style="
                            background-color: #f0f2f6;
                            padding: 20px;
                            border-radius: 10px;
                            border-left: 5px solid #ff6b6b;
                            margin: 10px 0;
                            color: #000000;
                            font-size: 16px;
                            line-height: 1.6;
                        ">
                            {reseña_generada}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Información adicional
                        palabras_generadas = len(reseña_generada.split()) - len(texto_inicial.split())
                        st.info(f"📊 Generado con: {modelo_seleccionado.upper()} | Temperatura: {temperatura} | Palabras añadidas: {palabras_generadas}")
                        
                        # Botón para generar otra versión
                        if st.button("🔄 Generar otra versión"):
                            st.rerun()
                            
                    else:
                        st.error("❌ Error generando reseña")
                        if reseña_generada:
                            st.write(reseña_generada)
            else:
                st.error("Por favor, escribe algo para comenzar la reseña")
    
    with col2:
        st.header("📊 Estado del Sistema")
        
        # Información del modelo
        if st.session_state.get('models_initialized', False):
            st.success("🤖 **Modelos:** ✅ Cargados")
        else:
            st.warning("🤖 **Modelos:** ❌ No cargados")
            
        modelo_display = "Transformer Modelo 1" if modelo_seleccionado == "model1" else "Transformer Modelo 2"
        st.info(f"**Modelo activo:** {modelo_display}")
        st.info(f"**Temperatura:** {temperatura}")
        st.info(f"**Longitud objetivo:** {longitud} palabras")
        
        # Ejemplos
        st.header("💡 Ejemplos de inicio")
        ejemplos = [
            "Esta película de acción",
            "El drama protagonizado por",
            "Una comedia que nos presenta",
            "Este thriller psicológico",
            "La película de terror más",
            "Un documental fascinante sobre",
            "Esta secuela supera",
            "El guión de esta película"
        ]
        
        for ejemplo in ejemplos:
            if st.button(f"📝 {ejemplo}", key=ejemplo):
                st.session_state.texto_inicial = ejemplo
                st.rerun()
        
        # Información adicional
        with st.expander("ℹ️ Información del Proyecto"):
            st.markdown("""
            **Generador de Reseñas de Películas**
            
            Este proyecto utiliza modelos Transformer para generar reseñas de películas 
            de forma automática basándose en un texto inicial.
            
            **Características:**
            - 🤖 Dos modelos Transformer diferentes
            - 🎛️ Control de creatividad mediante temperatura
            - 📏 Longitud de reseña configurable
            - 🎬 Entrenado con dataset IMDb (8000 tokens)
            
            **Tecnologías:**
            - TensorFlow
            - Streamlit
            - Transformers
            """)

if __name__ == "__main__":
    main()