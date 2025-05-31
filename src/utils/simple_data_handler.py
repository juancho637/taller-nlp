"""
Versión simplificada usando dataset IMDb de TensorFlow
NO requiere descargas manuales - más fácil para Mac
"""

import tensorflow as tf
from tensorflow import keras
from keras.layers import TextVectorization
import config

def create_simple_dataset():
    """
    Crear dataset usando IMDb integrado en TensorFlow (más fácil)
    """
    print("📊 Cargando dataset IMDb de TensorFlow...")
    
    # Cargar dataset IMDb directamente de TensorFlow
    (x_train, y_train), _ = keras.datasets.imdb.load_data()
    
    # Obtener el diccionario de palabras
    word_index = keras.datasets.imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    # Convertir números de vuelta a texto
    def decode_review(text):
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])
    
    # Convertir una muestra a texto
    print("🔄 Convirtiendo datos a texto...")
    text_data = []
    for i in range(min(config.SAMPLE_SIZE, len(x_train))):
        review_text = decode_review(x_train[i])
        text_data.append(review_text)
    
    # Crear dataset de TensorFlow
    dataset = tf.data.Dataset.from_tensor_slices(text_data)
    dataset = dataset.batch(config.BATCH_SIZE)
    
    print("✅ Dataset creado exitosamente")
    return dataset

def create_text_vectorizer(dataset):
    """
    Crea el vectorizador de texto
    """
    print("🔤 Preparando vectorizador de texto...")
    
    text_vectorization = TextVectorization(
        max_tokens=config.VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=config.SEQUENCE_LENGTH,
    )
    
    # Adaptar al dataset
    text_vectorization.adapt(dataset)
    
    print("✅ Vectorizador listo")
    return text_vectorization

def prepare_lm_dataset(dataset, text_vectorization):
    """
    Prepara el dataset para language modeling
    """
    print("🎯 Preparando dataset para entrenamiento...")
    
    def prepare_lm_inputs(text_batch):
        vectorized_sequences = text_vectorization(text_batch)
        x = vectorized_sequences[:, :-1]
        y = vectorized_sequences[:, 1:]
        return x, y

    lm_dataset = dataset.map(prepare_lm_inputs, num_parallel_calls=4)
    
    print("✅ Dataset de entrenamiento listo")
    return lm_dataset