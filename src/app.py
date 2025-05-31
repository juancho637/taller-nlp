"""
🎬 INTERFAZ WEB SIMPLE - UNA COLUMNA
===================================

Interfaz web súper simple para generar reseñas.
Ejecutar: streamlit run app.py
"""

import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pickle
import os

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="🎬 Generador de Reseñas",
    page_icon="🎬",
    layout="centered"  # UNA COLUMNA centrada
)

# ============================================================================
# CAPA PERSONALIZADA (DEBE ESTAR DEFINIDA PARA CARGAR MODELOS)
# ============================================================================

class PositionalEmbedding(layers.Layer):
    """Capa personalizada para embeddings posicionales"""
    
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        self.token_embeddings = layers.Embedding(vocab_size, embed_dim)
        self.position_embeddings = layers.Embedding(sequence_length, embed_dim)
    
    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        
        token_emb = self.token_embeddings(inputs)
        pos_emb = self.position_embeddings(positions)
        
        return token_emb + pos_emb
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Registrar la capa personalizada para que pueda ser cargada
keras.utils.get_custom_objects()['PositionalEmbedding'] = PositionalEmbedding

# ============================================================================
# FUNCIONES PARA MÉTRICAS
# ============================================================================

def get_model_info(model_name):
    """
    Obtiene información técnica de un modelo específico.
    
    Args:
        model_name: "1" o "2"
    """
    model_file = f"saved_models/movie_model_{model_name}.keras"
    
    if not os.path.exists(model_file):
        return None
    
    try:
        model = keras.models.load_model(model_file)
        
        # Información básica
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        
        # Arquitectura
        layers_info = []
        for i, layer in enumerate(model.layers):
            layer_info = {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'output_shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'N/A',
                'params': layer.count_params()
            }
            layers_info.append(layer_info)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'layers': layers_info,
            'model': model
        }
        
    except Exception as e:
        st.error(f"Error cargando modelo {model_name}: {str(e)}")
        return None

def show_model_metrics(model_name):
    """Mostrar métricas detalladas de un modelo específico"""
    model_info = get_model_info(model_name)
    
    if not model_info:
        st.error(f"❌ Modelo {model_name} no encontrado o no se pudo cargar")
        return
    
    # Configuración según el modelo
    config_info = {
        "1": {
            "nombre": "Transformer Simple",
            "capas_atencion": 1,
            "epocas_entrenamiento": 15,
            "learning_rate": 0.001,
            "fortalezas": ["Más estable", "Menos overfitting", "Entrenamiento rápido"],
            "uso_recomendado": "Texto coherente y predecible"
        },
        "2": {
            "nombre": "Transformer Doble",
            "capas_atencion": 2,
            "epocas_entrenamiento": 10,
            "learning_rate": 0.0005,
            "fortalezas": ["Más expresivo", "Mayor capacidad", "Texto más creativo"],
            "uso_recomendado": "Contenido original y variado"
        }
    }
    
    config = config_info[model_name]
    
    st.success(f"📊 **Métricas del Modelo {model_name}** - {config['nombre']}")
    
    # Métricas principales en una columna
    st.metric(
        label="🔢 Parámetros Totales",
        value=f"{model_info['total_params']:,}",
        help="Número total de parámetros entrenables del modelo"
    )
    
    st.metric(
        label="🎯 Capas de Atención",
        value=config['capas_atencion'],
        help="Número de capas Transformer de atención múltiple"
    )
    
    st.metric(
        label="📚 Épocas Entrenamiento",
        value=config['epocas_entrenamiento'],
        help="Número de épocas usadas durante el entrenamiento"
    )
    
    # Información técnica
    st.markdown("### 🔧 Configuración Técnica")
    
    st.markdown(f"""
    **⚙️ Hiperparámetros:**
    - Learning Rate: `{config['learning_rate']}`
    - Vocabulario: `5,000 tokens`
    - Secuencia máxima: `60 tokens`
    - Embedding dimension: `128`
    - Attention heads: `4`
    
    **🎯 Características:**
    - Parámetros entrenables: `{model_info['trainable_params']:,}`
    - Arquitectura: `{config['nombre']}`
    - Dataset: `IMDb reviews`
    - Tamaño muestra: `2,000 reseñas`
    """)
    
    # Fortalezas del modelo
    st.markdown("### ✨ Fortalezas del Modelo")
    strengths_text = " • ".join(config['fortalezas'])
    st.info(f"**{config['uso_recomendado']}**\n\n• {strengths_text}")
    
    # Arquitectura detallada
    with st.expander("🏗️ Ver Arquitectura Detallada"):
        st.markdown("#### Capas del Modelo:")
        
        for i, layer in enumerate(model_info['layers']):
            if layer['params'] > 0:  # Solo mostrar capas con parámetros
                st.markdown(f"""
                **Capa {i+1}: {layer['name']}**
                - Tipo: `{layer['type']}`
                - Forma salida: `{layer['output_shape']}`
                - Parámetros: `{layer['params']:,}`
                """)
    
    # Consejos de uso
    st.markdown("### 💡 Consejos de Uso")
    
    if model_name == "1":
        st.markdown("""
        **🎯 Mejor para:**
        - Usuarios principiantes
        - Reseñas coherentes y estables
        - Cuando necesitas resultados predecibles
        
        **🌡️ Temperatura recomendada:** 0.5 - 0.8
        **📏 Longitud ideal:** 30-50 palabras
        """)
    else:
        st.markdown("""
        **🎯 Mejor para:**
        - Usuarios avanzados
        - Contenido más creativo y original
        - Cuando quieres variedad en el texto
        
        **🌡️ Temperatura recomendada:** 0.7 - 1.2
        **📏 Longitud ideal:** 40-80 palabras
        """)

def compare_models():
    """Comparar ambos modelos lado a lado"""
    model1_info = get_model_info("1")
    model2_info = get_model_info("2")
    
    if not model1_info and not model2_info:
        st.error("❌ No se encontraron modelos entrenados")
        return
    elif not model1_info:
        st.warning("⚠️ Solo el Modelo 2 está disponible")
        show_model_metrics("2")
        return
    elif not model2_info:
        st.warning("⚠️ Solo el Modelo 1 está disponible")
        show_model_metrics("1")
        return
    
    st.success("⚖️ **Comparación de Modelos**")
    
    # Comparación en una columna
    st.markdown("### 🤖 Modelo 1 - Simple")
    st.markdown(f"""
    **📊 Métricas:**
    - Parámetros: `{model1_info['total_params']:,}`
    - Capas atención: `1`
    - Épocas: `15`
    - Learning Rate: `0.001`
    
    **✨ Fortalezas:**
    - ✅ Más estable
    - ✅ Menos overfitting  
    - ✅ Entrenamiento rápido
    - ✅ Resultados predecibles
    
    **🎯 Mejor para:**
    - Principiantes
    - Texto coherente
    - Uso general
    """)
    
    st.divider()
    
    st.markdown("### 🤖 Modelo 2 - Doble")
    st.markdown(f"""
    **📊 Métricas:**
    - Parámetros: `{model2_info['total_params']:,}`
    - Capas atención: `2`
    - Épocas: `10`
    - Learning Rate: `0.0005`
    
    **✨ Fortalezas:**
    - ✅ Más expresivo
    - ✅ Mayor capacidad
    - ✅ Texto más creativo
    - ✅ Mejor para contenido original
    
    **🎯 Mejor para:**
    - Usuarios avanzados
    - Contenido creativo
    - Variedad en el texto
    """)
    
    # Recomendaciones
    st.markdown("### 🎯 Recomendaciones de Uso")
    
    param_diff = model2_info['total_params'] - model1_info['total_params']
    param_percent = (param_diff / model1_info['total_params']) * 100
    
    st.info(f"""
    **📈 Diferencias clave:**
    - El Modelo 2 tiene **{param_diff:,} parámetros más** ({param_percent:.1f}% más complejo)
    - El Modelo 1 es **más estable** para principiantes
    - El Modelo 2 es **más creativo** pero requiere ajuste fino
    
    **💡 Consejo:** Empieza con el Modelo 1, luego experimenta con el Modelo 2
    """)
    
    # Tabla de configuraciones recomendadas
    st.markdown("### ⚙️ Configuraciones Recomendadas")
    
    st.markdown("""
    | Parámetro | Modelo 1 | Modelo 2 |
    |-----------|----------|----------|
    | **Temperatura** | 0.5 - 0.8 | 0.7 - 1.2 |
    | **Longitud** | 30-50 palabras | 40-80 palabras |
    | **Uso ideal** | Texto coherente | Texto creativo |
    """)

# ============================================================================
# FUNCIONES PARA CARGAR MODELOS
# ============================================================================

@st.cache_resource
def load_model(model_name):
    """
    Carga un modelo específico (1 o 2).
    
    Args:
        model_name: "1" o "2"
    """
    model_file = f"saved_models/movie_model_{model_name}.keras"
    
    if not os.path.exists(model_file):
        return None, f"No se encontró {model_file}. Entrena el modelo primero."
    
    try:
        model = keras.models.load_model(model_file)
        return model, None
    except Exception as e:
        return None, f"Error cargando {model_file}: {str(e)}"

@st.cache_resource
def load_vectorizer():
    """Carga el traductor de palabras"""
    vectorizer_file = "saved_models/text_vectorizer.pkl"
    
    if not os.path.exists(vectorizer_file):
        return None, None, f"No se encontró {vectorizer_file}. Entrena un modelo primero."
    
    try:
        with open(vectorizer_file, "rb") as f:
            vectorizer_data = pickle.load(f)
        
        # Recrear vectorizador de forma más simple
        vectorizer = keras.layers.TextVectorization(
            max_tokens=vectorizer_data['config']['max_tokens'],
            output_mode=vectorizer_data['config']['output_mode'],
            output_sequence_length=vectorizer_data['config']['output_sequence_length']
        )
        
        # Inicializar con datos dummy de forma más compatible
        dummy_texts = ["dummy text", "another text", "more text"]
        dummy_dataset = tf.data.Dataset.from_tensor_slices(dummy_texts).batch(1)
        vectorizer.adapt(dummy_dataset)
        
        # Cargar los pesos reales
        vectorizer.set_weights(vectorizer_data['weights'])
        
        # Crear diccionario de palabras
        vocabulary = vectorizer_data['vocabulary']
        id_to_word = {i: word for i, word in enumerate(vocabulary)}
        
        return vectorizer, id_to_word, None
        
    except Exception as e:
        return None, None, f"Error cargando vectorizer: {str(e)}"

# ============================================================================
# FUNCIÓN DE GENERACIÓN
# ============================================================================

def generate_review(model, vectorizer, id_to_word, prompt, temperature, max_words):
    """Genera una reseña con el modelo seleccionado"""
    try:
        current_text = prompt
        words_generated = 0
        
        # Mostrar progreso
        progress_placeholder = st.empty()
        
        for i in range(max_words):
            # Actualizar progreso
            progress_placeholder.text(f"🤖 Generando palabra {i+1}/{max_words}...")
            
            # Vectorizar texto actual
            input_ids = vectorizer([current_text])
            
            # Predecir siguiente palabra
            predictions = model(input_ids)
            last_word_probs = predictions[0, -1, :]
            
            # Aplicar temperatura para controlar creatividad
            last_word_probs = last_word_probs / temperature
            last_word_probs = tf.nn.softmax(last_word_probs)
            
            # Elegir siguiente palabra
            next_word_id = tf.random.categorical([last_word_probs], 1)[0, 0].numpy()
            
            # Convertir ID a palabra
            if next_word_id in id_to_word:
                next_word = id_to_word[next_word_id]
                
                # Filtrar palabras problemáticas
                if next_word in ['[UNK]', '', ' ', '<pad>', '<start>', '<end>']:
                    continue
                
                # Añadir nueva palabra
                current_text += " " + next_word
                words_generated += 1
                
                # Parar en punto final si ya tenemos suficiente texto
                if next_word.endswith('.') and words_generated > 10:
                    break
        
        # Limpiar progreso
        progress_placeholder.empty()
        
        return current_text, words_generated
        
    except Exception as e:
        return f"Error generando: {str(e)}", 0

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

def main():
    """Interfaz principal de una columna"""
    
    # TÍTULO
    st.title("🎬 Generador de Reseñas de Películas")
    st.markdown("*Genera reseñas automáticamente con Inteligencia Artificial*")
    
    # SELECCIÓN DE MODELO
    st.subheader("🤖 Selecciona el Modelo")
    
    model_choice = st.radio(
        "¿Qué modelo usar?",
        ["1", "2"],
        format_func=lambda x: f"Modelo {x} ({'Simple' if x == '1' else 'Doble'})",
        horizontal=True
    )
    
    st.info(f"""
    **Modelo {model_choice}:**
    {"• Una capa Transformer" if model_choice == "1" else "• Dos capas Transformer"}
    {"• Más estable y rápido" if model_choice == "1" else "• Más creativo y complejo"}
    {"• Recomendado para empezar" if model_choice == "1" else "• Para usuarios avanzados"}
    """)
    
    # CARGAR MODELO Y VECTORIZADOR
    model, model_error = load_model(model_choice)
    vectorizer, id_to_word, vec_error = load_vectorizer()
    
    # VERIFICAR ERRORES
    if model_error or vec_error:
        st.error(f"❌ {model_error or vec_error}")
        st.warning("💡 **Solución:** Ejecuta primero el entrenamiento:")
        st.code(f"python train.py --modelo {model_choice}")
        st.stop()
    
    st.success(f"✅ Modelo {model_choice} cargado correctamente")
    
    # SEPARADOR
    st.divider()
    
    # CONFIGURACIÓN DE GENERACIÓN
    st.subheader("✍️ Escribe tu Reseña")
    
    # Prompt del usuario
    prompt = st.text_area(
        "¿Cómo quieres que empiece la reseña?",
        value="This movie is",
        height=100,
        help="Escribe las primeras palabras y la IA completará el resto",
        placeholder="Ejemplo: 'This movie is'"
    )
    
    # Controles
    temperature = st.slider(
        "🌡️ Creatividad",
        min_value=0.1,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="0.1 = muy conservador y predecible\n2.0 = muy creativo y arriesgado"
    )
    
    max_words = st.slider(
        "📏 Longitud",
        min_value=10,
        max_value=100,
        value=50,
        step=5,
        help="Número máximo de palabras a generar"
    )
    
    # BOTÓN DE GENERACIÓN
    if st.button("🚀 Generar Reseña", type="primary", use_container_width=True):
        if not prompt.strip():
            st.error("❌ Por favor escribe algo para empezar la reseña")
        else:
            # Generar reseña
            with st.spinner("🤖 La IA está escribiendo tu reseña..."):
                generated_text, words_added = generate_review(
                    model, vectorizer, id_to_word, 
                    prompt, temperature, max_words
                )
            
            # Mostrar resultado
            if words_added > 0:
                st.success("🎉 ¡Reseña generada exitosamente!")
                
                # Mostrar la reseña con estilo
                st.markdown("### 📝 Tu Reseña:")
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 25px;
                    border-radius: 15px;
                    color: white;
                    font-size: 18px;
                    line-height: 1.8;
                    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                    margin: 20px 0;
                    border: 2px solid rgba(255,255,255,0.1);
                ">
                    <strong>"{generated_text}"</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Estadísticas
                total_words = len(generated_text.split())
                st.markdown(f"""
                **📊 Estadísticas:**
                - 🔤 Palabras añadidas: {words_added}
                - 📏 Total de palabras: {total_words}
                - 🌡️ Creatividad usada: {temperature}
                - 🤖 Modelo: {model_choice} ({'Simple' if model_choice == '1' else 'Doble'})
                """)
                
                # Botón para generar otra
                if st.button("🔄 Generar otra versión", use_container_width=True):
                    st.rerun()
            else:
                st.error("❌ No se pudo generar la reseña. Intenta ajustar los parámetros.")
    
    # SEPARADOR
    st.divider()
    
    # MÉTRICAS DE LOS MODELOS
    st.subheader("📊 Métricas de los Modelos")
    
    if st.button("📈 Ver Métricas Modelo 1", use_container_width=True):
        show_model_metrics("1")
    
    if st.button("📈 Ver Métricas Modelo 2", use_container_width=True):
        show_model_metrics("2")
    
    # Botón para comparar ambos modelos
    if st.button("⚖️ Comparar Ambos Modelos", type="secondary", use_container_width=True):
        compare_models()
    
    # INFORMACIÓN DEL PROYECTO
    with st.expander("ℹ️ ¿Cómo funciona este proyecto?"):
        st.markdown(f"""
        ### 🧠 Tecnología
        
        **Modelo Transformer {model_choice}:**
        - {"🔹 Una capa de atención (más simple)" if model_choice == "1" else "🔹 Dos capas de atención (más complejo)"}
        - 🔹 Entrenado con reseñas reales de IMDb
        - 🔹 Vocabulario de 5,000 palabras cinematográficas
        
        ### 🎯 ¿Cómo genera texto?
        1. **Lee** el inicio que escribes
        2. **Analiza** patrones de las reseñas que conoce
        3. **Predice** qué palabra podría venir después
        4. **Repite** el proceso hasta formar una reseña completa
        
        ### 🛠️ Tecnologías usadas
        - **TensorFlow**: Framework de IA
        - **Transformers**: Arquitectura del modelo
        - **Dataset IMDb**: Reseñas reales para entrenamiento
        - **Streamlit**: Esta interfaz web
        
        ### 🎛️ Controles
        - **Creatividad baja (0.1-0.5)**: Texto más predecible y coherente
        - **Creatividad media (0.6-1.0)**: Balance entre coherencia y originalidad
        - **Creatividad alta (1.1-2.0)**: Texto más arriesgado y creativo
        """)

if __name__ == "__main__":
    main()