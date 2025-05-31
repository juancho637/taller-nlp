"""
Generador de texto con lógica mejorada para coherencia
VERSIÓN MEJORADA - Genera texto más coherente y legible
"""

import tensorflow as tf
from keras.layers import TextVectorization
import numpy as np
import config
import os

class ReviewGenerator:
    """
    Clase para cargar modelos y generar reseñas coherentes
    """
    
    def __init__(self):
        self.model1 = None
        self.model2 = None
        self.text_vectorization = None
        self.tokens_index = None
        self.models_loaded = False
    
    def load_models(self):
        """
        Cargar los modelos entrenados y el vectorizador
        """
        try:
            print("📂 Cargando modelos...")
            
            # Cargar modelos con extensión .keras
            model1_path = f"{config.MODEL_SAVE_PATH}/transformer_model_1.keras"
            model2_path = f"{config.MODEL_SAVE_PATH}/transformer_model_2.keras"
            
            if os.path.exists(model1_path) and os.path.exists(model2_path):
                # Importar clases personalizadas para cargar
                from models.transformer_models import PositionalEmbedding, TransformerDecoder
                
                custom_objects = {
                    'PositionalEmbedding': PositionalEmbedding,
                    'TransformerDecoder': TransformerDecoder
                }
                
                self.model1 = tf.keras.models.load_model(model1_path, custom_objects=custom_objects)
                self.model2 = tf.keras.models.load_model(model2_path, custom_objects=custom_objects)
                
                # Cargar vectorizador
                self._load_vectorizer()
                
                self.models_loaded = True
                print("✅ Modelos cargados exitosamente")
                return True
            else:
                print("❌ Modelos no encontrados. Ejecuta solucion_definitiva.py primero")
                return False
                
        except Exception as e:
            print(f"❌ Error cargando modelos: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_real_dataset(self):
        """
        Crear dataset real para adaptar el vectorizador
        """
        try:
            # Opción 1: Usar dataset local si existe
            if os.path.exists("../aclImdb"):
                print("📁 Usando dataset aclImdb local...")
                from tensorflow import keras
                dataset = keras.utils.text_dataset_from_directory(
                    directory="../aclImdb", 
                    label_mode=None, 
                    batch_size=32
                )
                # Limpiar <br /> más agresivamente
                dataset = dataset.map(lambda x: tf.strings.regex_replace(x, "<br />", " "))
                dataset = dataset.map(lambda x: tf.strings.regex_replace(x, "br", ""))
                dataset = dataset.map(lambda x: tf.strings.regex_replace(x, r'\s+', ' '))
                # Solo tomar una muestra pequeña para vocabulario
                dataset = dataset.take(50)
                return dataset
            else:
                # Opción 2: Usar el método de simple_data_handler
                print("📊 Usando dataset de TensorFlow...")
                from utils.simple_data_handler import create_simple_dataset
                return create_simple_dataset()
                
        except Exception as e:
            print(f"⚠️ Error creando dataset real: {e}")
            # Último recurso: dataset mínimo con palabras de cine
            print("🆘 Usando dataset de emergencia...")
            emergency_data = [
                "this movie is great and amazing with excellent acting",
                "the film was terrible and disappointing with bad plot", 
                "excellent cinematography and brilliant direction make this worth watching",
                "worst movie ever made with poor acting and boring story",
                "fantastic thriller with great suspense and outstanding performances",
                "disappointing sequel to a good film with weak script",
                "amazing visual effects and compelling story create masterpiece",
                "boring drama with predictable plot and mediocre acting"
            ]
            return tf.data.Dataset.from_tensor_slices(emergency_data).batch(4)
    
    def _load_vectorizer(self):
        """
        Cargar y reconstruir el vectorizador de texto CORRECTAMENTE
        """
        try:
            # Cargar configuración y pesos del vectorizador
            vectorizer_config = np.load(f"{config.MODEL_SAVE_PATH}/vectorizer_config.npy", allow_pickle=True).item()
            vectorizer_weights = np.load(f"{config.MODEL_SAVE_PATH}/vectorizer_weights.npy", allow_pickle=True)
            
            print(f"🔤 Cargando vectorizador con configuración: max_tokens={vectorizer_config['max_tokens']}")
            
            # Recrear el vectorizador con la configuración original
            self.text_vectorization = TextVectorization(
                max_tokens=vectorizer_config['max_tokens'],
                output_mode=vectorizer_config['output_mode'],
                output_sequence_length=vectorizer_config['output_sequence_length']
            )
            
            # CORREGIDO: Adaptar con dataset REAL, no dummy
            print("🎯 Adaptando vectorizador con dataset real...")
            real_dataset = self._create_real_dataset()
            self.text_vectorization.adapt(real_dataset)
            
            # Cargar pesos solo si existen
            if len(vectorizer_weights) > 0:
                self.text_vectorization.set_weights(vectorizer_weights)
                print("✅ Pesos del vectorizador cargados")
            else:
                print("⚠️ No hay pesos guardados, usando vocabulario adaptado")
            
            # Crear diccionario de tokens
            vocabulary = self.text_vectorization.get_vocabulary()
            self.tokens_index = dict(enumerate(vocabulary))
            
            print(f"✅ Vocabulario cargado: {len(vocabulary)} tokens")
            print(f"🔤 Primeros 10 tokens: {list(vocabulary[:10])}")
            
            # Verificar que tenemos palabras reales
            movie_words = [w for w in vocabulary if w in ['movie', 'film', 'good', 'bad', 'great', 'excellent', 'terrible', 'amazing']]
            print(f"🎬 Palabras de cine encontradas: {movie_words}")
            
            print(f"📊 Rango de IDs válidos: 0 - {len(vocabulary)-1}")
            
        except Exception as e:
            print(f"❌ Error cargando vectorizador: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def sample_next_token_inteligente(self, predictions, temperature=1.0, avoid_tokens=None):
        """
        Muestrear token con lógica mejorada para evitar repetición
        """
        predictions = np.asarray(predictions).astype("float64")
        
        if len(predictions) == 0:
            return 1
        
        # Crear máscara para evitar tokens problemáticos
        if avoid_tokens:
            for token_id in avoid_tokens:
                if token_id < len(predictions):
                    predictions[token_id] *= 0.1  # Reducir probabilidad dramáticamente
        
        # Aplicar temperatura
        predictions = np.log(predictions + 1e-8) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)
        
        # Ajustar al vocabulario
        max_vocab_id = len(self.tokens_index) - 1
        if len(predictions) > max_vocab_id + 1:
            predictions = predictions[:max_vocab_id + 1]
        
        # Muestrear con top-k para mejor calidad
        k = min(50, len(predictions))  # Solo considerar top 50 tokens
        top_k_indices = np.argsort(predictions)[-k:]
        top_k_probs = predictions[top_k_indices]
        top_k_probs = top_k_probs / np.sum(top_k_probs)
        
        try:
            sampled_idx = np.random.choice(len(top_k_probs), p=top_k_probs)
            sampled_id = top_k_indices[sampled_idx]
            
            if sampled_id > max_vocab_id:
                sampled_id = np.random.randint(1, min(max_vocab_id + 1, 100))
                
            return sampled_id
        except Exception as e:
            print(f"❌ Error en sampling: {e}")
            return 1
    
    def generate_review(self, prompt, model_choice="model1", temperature=0.7, max_length=100):
        """
        Generar una reseña coherente basada en el prompt inicial
        """
        if not self.models_loaded:
            return "❌ Error: Modelos no cargados. Ejecuta el entrenamiento primero."
        
        try:
            # Seleccionar modelo
            model = self.model1 if model_choice == "model1" else self.model2
            
            # Ajustar temperatura para mejor coherencia
            if temperature > 1.2:
                temperature = 1.2  # Limitar temperatura alta
            if temperature < 0.3:
                temperature = 0.3  # Temperatura mínima para variedad
            
            # Preparar el prompt
            sentence = prompt.strip()
            generated_tokens = 0
            
            print(f"🎬 Generando reseña coherente con {model_choice}...")
            print(f"🌡️ Temperatura ajustada: {temperature}")
            
            # Tracking para evitar repetición
            recent_words = []
            word_count = {}
            
            # Tokens problemáticos a evitar
            problematic_tokens = set()
            vocabulary = self.text_vectorization.get_vocabulary()
            for i, token in enumerate(vocabulary):
                if token in ['br', '<br>', '[UNK]', '', ' ', 'the', 'and', 'a', 'an', 'is', 'on', 'it', 'to']:
                    if token in ['br', '<br>', '[UNK]', '']:  # Evitar completamente
                        problematic_tokens.add(i)
            
            # Generar tokens con lógica mejorada
            for i in range(max_length):
                # Vectorizar la frase actual
                tokenized_sentence = self.text_vectorization([sentence])
                
                # Predecir siguiente palabra
                predictions = model(tokenized_sentence)
                last_token_predictions = predictions[0, -1, :]
                
                # Crear lista de tokens a evitar basada en repetición reciente
                avoid_tokens = list(problematic_tokens)
                
                # Evitar palabras muy repetidas
                for word, count in word_count.items():
                    if count > 2:  # Si una palabra aparece más de 2 veces
                        word_tokens = self.text_vectorization([word])
                        if word_tokens.shape[1] > 0:
                            token_id = word_tokens[0, 0].numpy()
                            avoid_tokens.append(token_id)
                
                # Muestrear con lógica inteligente
                next_token_id = self.sample_next_token_inteligente(
                    last_token_predictions, 
                    temperature, 
                    avoid_tokens
                )
                
                # Convertir ID a palabra
                if next_token_id in self.tokens_index:
                    next_word = self.tokens_index[next_token_id]
                    
                    # Filtros de calidad
                    if len(next_word) == 0 or next_word in ['br', '<br>', '[UNK]']:
                        continue
                    
                    # Evitar repetición inmediata
                    if len(recent_words) > 0 and next_word == recent_words[-1]:
                        continue
                    
                    # Evitar demasiadas palabras comunes seguidas
                    if next_word in ['the', 'and', 'a', 'an', 'is', 'on', 'it', 'to']:
                        if len(recent_words) > 0 and recent_words[-1] in ['the', 'and', 'a', 'an', 'is', 'on', 'it', 'to']:
                            continue
                    
                    # Añadir palabra
                    sentence += " " + next_word
                    generated_tokens += 1
                    
                    # Actualizar tracking
                    recent_words.append(next_word)
                    if len(recent_words) > 5:  # Solo mantener las últimas 5 palabras
                        recent_words.pop(0)
                    
                    word_count[next_word] = word_count.get(next_word, 0) + 1
                    
                    # Condiciones de parada inteligentes
                    if next_word.endswith('.') and generated_tokens > 20:
                        print("🔚 Terminando en punto final")
                        break
                    
                    if generated_tokens > 30 and next_word in ['.', '!', '?', 'movie', 'film']:
                        break
                        
                    # Parar si detectamos bucle (misma palabra muchas veces)
                    if word_count.get(next_word, 0) > 3:
                        print("🔄 Detectado bucle de palabras, terminando")
                        break
                        
                else:
                    print(f"⚠️ Token ID {next_token_id} no encontrado")
                    break
            
            print(f"✅ Generación completada. Tokens generados: {generated_tokens}")
            
            # Post-procesamiento para limpiar el texto
            sentence = self._clean_generated_text(sentence)
            
            return sentence
            
        except Exception as e:
            print(f"❌ Error generando reseña: {e}")
            return f"Error generando reseña: {str(e)}"
    
    def _clean_generated_text(self, text):
        """
        Limpiar el texto generado para mejor legibilidad
        """
        import re
        
        # Remover tokens problemáticos
        text = re.sub(r'\bbr\b', '', text)
        text = re.sub(r'<br/?>', '', text)
        
        # Limpiar espacios múltiples
        text = re.sub(r'\s+', ' ', text)
        
        # Capitalizar primera letra después de puntos
        text = re.sub(r'([.!?]\s*)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
        
        # Asegurar que la primera letra esté en mayúscula
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        return text.strip()

# Instancia global del generador
review_generator = ReviewGenerator()

def initialize_generator():
    """
    Inicializar el generador de reseñas
    """
    return review_generator.load_models()

def generate_movie_review(prompt, model_choice="model1", temperature=0.7, length=100):
    """
    Función wrapper para generar reseñas coherentes
    """
    if not review_generator.models_loaded:
        success = initialize_generator()
        if not success:
            return "❌ No se pudieron cargar los modelos. Asegúrate de haber ejecutado solucion_definitiva.py primero."
    
    return review_generator.generate_review(prompt, model_choice, temperature, length)