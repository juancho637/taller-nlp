"""
ENTRENAMIENTO SEPARADO POR MODELOS
Permite entrenar cada modelo independientemente con configuraciones específicas
"""

import os
import argparse
import tensorflow as tf
from tensorflow import keras
from keras.layers import TextVectorization
import config
from models.transformer_models import create_transformer_model_1, create_transformer_model_2
import numpy as np
import shutil

def limpiar_modelos():
    """
    Limpiar modelos anteriores (opcional)
    """
    print("🧹 Limpiando modelos anteriores...")
    
    if os.path.exists(config.MODEL_SAVE_PATH):
        shutil.rmtree(config.MODEL_SAVE_PATH)
    
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    print("✅ Directorio limpio creado")

def crear_dataset_optimizado():
    """
    Crear dataset con preprocesamiento mejorado
    """
    print("📊 Creando dataset optimizado...")
    
    try:
        if os.path.exists("../aclImdb"):
            print("📁 Usando dataset aclImdb local...")
            dataset = keras.utils.text_dataset_from_directory(
                directory="../aclImdb", 
                label_mode=None, 
                batch_size=config.BATCH_SIZE
            )
            
            # Limpieza mejorada
            def limpiar_texto(text):
                text = tf.strings.regex_replace(text, "<br />", " ")
                text = tf.strings.regex_replace(text, "<br/>", " ")
                text = tf.strings.regex_replace(text, "<.*?>", " ")
                text = tf.strings.regex_replace(text, r"[^\w\s\.,!?']", " ")
                text = tf.strings.regex_replace(text, r"\s+", " ")
                text = tf.strings.lower(text)
                return text
            
            dataset = dataset.map(limpiar_texto)
            
            # Filtrar por longitud
            def filtrar_longitud(text):
                length = tf.strings.length(text)
                return tf.logical_and(length > 50, length < 1000)
            
            dataset = dataset.filter(filtrar_longitud)
            sample_dataset = dataset.take(config.SAMPLE_SIZE // config.BATCH_SIZE)
            
        else:
            print("📊 Usando dataset de TensorFlow...")
            (x_train, y_train), _ = keras.datasets.imdb.load_data()
            
            word_index = keras.datasets.imdb.get_word_index()
            reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
            
            def decode_review(text):
                words = []
                for i in text:
                    word = reverse_word_index.get(i - 3, '')
                    if word and len(word) > 1:
                        words.append(word)
                return ' '.join(words)
            
            text_data = []
            for i in range(min(config.SAMPLE_SIZE, len(x_train))):
                review_text = decode_review(x_train[i])
                if len(review_text) > 50:
                    text_data.append(review_text)
            
            sample_dataset = tf.data.Dataset.from_tensor_slices(text_data)
            sample_dataset = sample_dataset.batch(config.BATCH_SIZE)
        
        print("✅ Dataset optimizado creado")
        return sample_dataset
        
    except Exception as e:
        print(f"❌ Error creando dataset: {e}")
        raise

def crear_o_cargar_vectorizador(dataset, force_new=False):
    """
    Crear nuevo vectorizador o cargar existente
    """
    vectorizer_path = f"{config.MODEL_SAVE_PATH}/vectorizer_config.npy"
    weights_path = f"{config.MODEL_SAVE_PATH}/vectorizer_weights.npy"
    
    if not force_new and os.path.exists(vectorizer_path) and os.path.exists(weights_path):
        print("📂 Cargando vectorizador existente...")
        try:
            vectorizer_config = np.load(vectorizer_path, allow_pickle=True).item()
            vectorizer_weights = np.load(weights_path, allow_pickle=True)
            
            text_vectorization = TextVectorization(
                max_tokens=vectorizer_config['max_tokens'],
                output_mode=vectorizer_config['output_mode'],
                output_sequence_length=vectorizer_config['output_sequence_length']
            )
            
            # Adaptar con dataset actual
            text_vectorization.adapt(dataset)
            
            # Cargar pesos si existen
            if len(vectorizer_weights) > 0:
                text_vectorization.set_weights(vectorizer_weights)
            
            vocabulary = text_vectorization.get_vocabulary()
            print(f"✅ Vectorizador cargado: {len(vocabulary)} tokens")
            return text_vectorization
            
        except Exception as e:
            print(f"⚠️ Error cargando vectorizador existente: {e}")
            print("🔄 Creando vectorizador nuevo...")
    
    # Crear vectorizador nuevo
    print("🔤 Creando vectorizador nuevo...")
    text_vectorization = TextVectorization(
        max_tokens=config.VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=config.SEQUENCE_LENGTH,
        standardize='lower_and_strip_punctuation',
        split='whitespace'
    )
    
    text_vectorization.adapt(dataset)
    
    # Guardar vectorizador
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    vectorizer_config = text_vectorization.get_config()
    vectorizer_weights = text_vectorization.get_weights()
    
    np.save(vectorizer_path, vectorizer_config)
    np.save(weights_path, vectorizer_weights, allow_pickle=True)
    
    vocabulary = text_vectorization.get_vocabulary()
    print(f"✅ Vectorizador nuevo creado: {len(vocabulary)} tokens")
    
    # Mostrar palabras de cine
    movie_words = [w for w in vocabulary if w in [
        'movie', 'film', 'good', 'bad', 'great', 'excellent', 'terrible', 
        'amazing', 'awful', 'brilliant', 'worst', 'best', 'story', 'plot',
        'acting', 'actor', 'actress', 'director', 'scene', 'character'
    ]]
    print(f"🎬 Palabras de cine encontradas ({len(movie_words)}): {movie_words}")
    
    return text_vectorization

def preparar_datos_entrenamiento(dataset, text_vectorization):
    """
    Preparar datos para entrenamiento
    """
    def prepare_lm_inputs(text_batch):
        vectorized_sequences = text_vectorization(text_batch)
        x = vectorized_sequences[:, :-1]
        y = vectorized_sequences[:, 1:]
        return x, y

    lm_dataset = dataset.map(prepare_lm_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    lm_dataset = lm_dataset.prefetch(tf.data.AUTOTUNE)
    
    return lm_dataset

def entrenar_modelo_1(lm_dataset, epochs=40):
    """
    Entrenar específicamente el Modelo 1
    """
    print(f"\n🏋️ ENTRENANDO MODELO 1 POR {epochs} ÉPOCAS")
    print("=" * 60)
    
    # Crear modelo
    model1 = create_transformer_model_1()
    print(f"📊 Modelo 1: {model1.count_params():,} parámetros")
    
    # Callbacks optimizados para Modelo 1
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='loss', 
            patience=15,  # Más paciencia para Modelo 1
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=7,
            min_lr=0.0001,
            verbose=1
        )
    ]
    
    # Entrenar
    history = model1.fit(
        lm_dataset, 
        epochs=epochs, 
        verbose=1,
        callbacks=callbacks
    )
    
    # Guardar
    model1.save(f"{config.MODEL_SAVE_PATH}/transformer_model_1.keras")
    
    # Estadísticas
    best_loss = min(history.history['loss'])
    print(f"\n✅ Modelo 1 completado - Mejor loss: {best_loss:.4f}")
    
    return model1, history

def entrenar_modelo_2(lm_dataset, epochs=25):
    """
    Entrenar específicamente el Modelo 2 (con menos épocas para evitar overfitting)
    """
    print(f"\n🏋️ ENTRENANDO MODELO 2 POR {epochs} ÉPOCAS")
    print("=" * 60)
    
    # Crear modelo
    model2 = create_transformer_model_2()
    print(f"📊 Modelo 2: {model2.count_params():,} parámetros")
    
    # Callbacks MÁS RESTRICTIVOS para Modelo 2 (evitar overfitting)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='loss', 
            patience=8,   # MENOS paciencia para Modelo 2
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.7,   # Reducción más suave
            patience=4,   # Más rápido
            min_lr=0.0002, # LR mínimo más alto
            verbose=1
        )
    ]
    
    # Entrenar con learning rate más bajo para Modelo 2
    model2.compile(
        loss="sparse_categorical_crossentropy", 
        optimizer=keras.optimizers.RMSprop(learning_rate=0.0003)  # MÁS BAJO
    )
    
    # Entrenar
    history = model2.fit(
        lm_dataset, 
        epochs=epochs, 
        verbose=1,
        callbacks=callbacks
    )
    
    # Guardar
    model2.save(f"{config.MODEL_SAVE_PATH}/transformer_model_2.keras")
    
    # Estadísticas
    best_loss = min(history.history['loss'])
    print(f"\n✅ Modelo 2 completado - Mejor loss: {best_loss:.4f}")
    
    return model2, history

def main():
    """
    Función principal con argumentos de línea de comandos
    """
    parser = argparse.ArgumentParser(description='Entrenamiento separado de modelos Transformer')
    
    parser.add_argument('--modelo', choices=['1', '2', 'ambos'], required=True,
                       help='Qué modelo entrenar: 1, 2, o ambos')
    parser.add_argument('--epocas1', type=int, default=40,
                       help='Épocas para Modelo 1 (default: 40)')
    parser.add_argument('--epocas2', type=int, default=25,
                       help='Épocas para Modelo 2 (default: 25)')
    parser.add_argument('--limpiar', action='store_true',
                       help='Limpiar modelos anteriores antes de entrenar')
    parser.add_argument('--nuevo-vectorizador', action='store_true',
                       help='Crear nuevo vectorizador (ignorar existente)')
    
    args = parser.parse_args()
    
    print("🚀 ENTRENAMIENTO SEPARADO DE MODELOS TRANSFORMER")
    print("=" * 60)
    print(f"📋 Configuración:")
    print(f"   - Modelo(s) a entrenar: {args.modelo}")
    print(f"   - Épocas Modelo 1: {args.epocas1}")
    print(f"   - Épocas Modelo 2: {args.epocas2}")
    print(f"   - Limpiar modelos: {args.limpiar}")
    print(f"   - Nuevo vectorizador: {args.nuevo_vectorizador}")
    print("=" * 60)
    
    try:
        # Limpiar si se solicita
        if args.limpiar:
            limpiar_modelos()
        else:
            os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        
        # Crear dataset
        dataset = crear_dataset_optimizado()
        
        # Crear o cargar vectorizador
        text_vectorization = crear_o_cargar_vectorizador(dataset, args.nuevo_vectorizador)
        
        # Preparar datos
        lm_dataset = preparar_datos_entrenamiento(dataset, text_vectorization)
        
        # Entrenar según la selección
        if args.modelo == '1':
            entrenar_modelo_1(lm_dataset, args.epocas1)
        elif args.modelo == '2':
            entrenar_modelo_2(lm_dataset, args.epocas2)
        elif args.modelo == 'ambos':
            entrenar_modelo_1(lm_dataset, args.epocas1)
            entrenar_modelo_2(lm_dataset, args.epocas2)
        
        print("\n🎉 ¡ENTRENAMIENTO COMPLETADO!")
        print("🚀 Ejecuta: uv run streamlit run app.py")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
