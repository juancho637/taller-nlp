"""
🎬 INTERFAZ WEB SIMPLE - GENERADOR DE RESEÑAS
============================================

Interfaz web simplificada para generar reseñas.
Solo carga modelos, muestra summary y genera texto.

Ejecutar: streamlit run app.py
"""

import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pickle
import os
import io
import sys

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="🎬 Generador de Reseñas",
    page_icon="🎬",
    layout="centered"
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

# Registrar la capa personalizada
keras.utils.get_custom_objects()['PositionalEmbedding'] = PositionalEmbedding

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
    """Carga el vectorizador de texto"""
    vectorizer_file = "saved_models/text_vectorizer.pkl"
    
    if not os.path.exists(vectorizer_file):
        return None, None, f"No se encontró {vectorizer_file}. Entrena un modelo primero."
    
    try:
        with open(vectorizer_file, "rb") as f:
            vectorizer_data = pickle.load(f)
        
        # Recrear vectorizador
        vectorizer = keras.layers.TextVectorization(
            max_tokens=vectorizer_data['config']['max_tokens'],
            output_mode=vectorizer_data['config']['output_mode'],
            output_sequence_length=vectorizer_data['config']['output_sequence_length']
        )
        
        # Inicializar con datos dummy
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
# FUNCIÓN PARA OBTENER SUMMARY DEL MODELO
# ============================================================================

def get_model_summary(model):
    """
    Obtiene el summary del modelo como string.
    
    Args:
        model: Modelo de TensorFlow/Keras cargado
        
    Returns:
        str: Summary del modelo
    """
    # Capturar el summary en un string
    stringio = io.StringIO()
    model.summary(print_fn=lambda x: stringio.write(x + '\n'))
    summary_string = stringio.getvalue()
    stringio.close()
    
    return summary_string

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
    """Interfaz principal simplificada"""
    
    # TÍTULO
    st.title("🎬 Generador de Reseñas de Películas")
    st.markdown("*Genera reseñas automáticamente con Inteligencia Artificial*")
    
    # SELECCIÓN DE MODELO
    st.subheader("🤖 Selecciona el Modelo")
    
    model_choice = st.radio(
        "¿Qué modelo usar?",
        ["1", "2"],
        format_func=lambda x: f"Modelo {x}",
        horizontal=True
    )
    
    # CARGAR MODELO Y VECTORIZADOR
    model, model_error = load_model(model_choice)
    vectorizer, id_to_word, vec_error = load_vectorizer()
    
    # VERIFICAR ERRORES
    if model_error or vec_error:
        st.error(f"❌ {model_error or vec_error}")
        st.warning("💡 **Solución:** Ejecuta primero el entrenamiento:")
        st.code(f"python train.py --model {model_choice}")
        st.stop()
    
    st.success(f"✅ Modelo {model_choice} cargado correctamente")
    
    # MOSTRAR SUMMARY DEL MODELO
    st.subheader(f"📊 Summary del Modelo {model_choice}")
    
    if st.button(f"📋 Mostrar Detalles del Modelo {model_choice}", use_container_width=True):
        try:
            # Obtener información básica del modelo
            total_params = model.count_params()
            trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
            
            # Mostrar información básica
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📊 Parámetros Totales", f"{total_params:,}")
            with col2:
                st.metric("🎯 Parámetros Entrenables", f"{trainable_params:,}")
            
            # Mostrar summary completo
            st.markdown("### 🏗️ Arquitectura del Modelo")
            summary_text = get_model_summary(model)
            st.code(summary_text, language="text")
            
        except Exception as e:
            st.error(f"Error obteniendo summary: {str(e)}")
    
    # SEPARADOR
    st.divider()
    
    # CONFIGURACIÓN DE GENERACIÓN
    st.subheader("✍️ Genera tu Reseña")
    
    # Prompt del usuario
    prompt = st.text_area(
        "¿Cómo quieres que empiece la reseña?",
        value="This movie is",
        height=100,
        help="Escribe las primeras palabras y la IA completará el resto",
        placeholder="Ejemplo: 'This movie is'"
    )
    
    # Controles
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.slider(
            "🌡️ Creatividad",
            min_value=0.1,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="0.1 = conservador, 2.0 = muy creativo"
        )
    
    with col2:
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
                
                # Mostrar la reseña
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
                
                # Estadísticas simples
                total_words = len(generated_text.split())
                st.markdown(f"""
                **📊 Estadísticas:**
                - 🔤 Palabras añadidas: {words_added}
                - 📏 Total de palabras: {total_words}
                - 🌡️ Creatividad usada: {temperature}
                - 🤖 Modelo usado: {model_choice}
                """)
                
            else:
                st.error("❌ No se pudo generar la reseña. Intenta ajustar los parámetros.")

if __name__ == "__main__":
    main()