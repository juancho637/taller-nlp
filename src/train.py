"""
🏋️ ENTRENAMIENTO DE MODELOS TRANSFORMER
=======================================

Script para entrenar los modelos por separado.

Uso:
    python train.py --modelo 1    # Solo Modelo 1
    python train.py --modelo 2    # Solo Modelo 2
    python train.py --modelo all  # Ambos modelos
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import argparse
import pickle
import os

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

class Config:
    """Configuración para el entrenamiento"""
    # Parámetros del modelo
    VOCAB_SIZE = 5000           # Vocabulario de palabras
    SEQUENCE_LENGTH = 60        # Longitud máxima de secuencia
    EMBED_DIM = 128            # Dimensión de embeddings
    NUM_HEADS = 4              # Cabezas de atención
    LATENT_DIM = 256           # Dimensión de la capa densa
    
    # Entrenamiento
    BATCH_SIZE = 32
    SAMPLE_SIZE = 2000         # Muestras del dataset
    
    # Configuración específica por modelo
    MODEL1_EPOCHS = 15         # Modelo 1: Menos épocas, más estable
    MODEL1_LR = 0.001         # Learning rate normal
    
    MODEL2_EPOCHS = 10         # Modelo 2: Pocas épocas para evitar overfitting
    MODEL2_LR = 0.0005        # Learning rate más bajo

# ============================================================================
# CAPA PERSONALIZADA PARA EMBEDDINGS POSICIONALES
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

# Registrar la capa personalizada de forma manual
keras.utils.get_custom_objects()['PositionalEmbedding'] = PositionalEmbedding

# ============================================================================
# MODELO 1: TRANSFORMER SIMPLE (MÁS ESTABLE)
# ============================================================================

def create_model_1():
    """
    Modelo 1: Una sola capa Transformer
    - Más estable y fácil de entrenar
    - Menos parámetros, menos overfitting
    - Recomendado para empezar
    """
    print("🏗️ Creando Modelo 1 (Simple)...")
    
    inputs = keras.Input(shape=(None,), dtype="int32")
    
    # Embeddings de tokens y posiciones usando la capa personalizada
    x = PositionalEmbedding(
        sequence_length=Config.SEQUENCE_LENGTH,
        vocab_size=Config.VOCAB_SIZE,
        embed_dim=Config.EMBED_DIM
    )(inputs)
    
    # UNA SOLA capa de atención
    attention = layers.MultiHeadAttention(
        num_heads=Config.NUM_HEADS, 
        key_dim=Config.EMBED_DIM
    )(x, x)
    x = layers.LayerNormalization()(x + attention)
    
    # Red neuronal feed-forward
    ffn = keras.Sequential([
        layers.Dense(Config.LATENT_DIM, activation="relu"),
        layers.Dense(Config.EMBED_DIM)
    ])(x)
    x = layers.LayerNormalization()(x + ffn)
    
    # Capa de salida
    outputs = layers.Dense(Config.VOCAB_SIZE, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(Config.MODEL1_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print(f"✅ Modelo 1 creado: {model.count_params():,} parámetros")
    return model

# ============================================================================
# MODELO 2: TRANSFORMER DOBLE (MÁS COMPLEJO)
# ============================================================================

def create_model_2():
    """
    Modelo 2: Dos capas Transformer
    - Más expresivo pero puede hacer overfitting
    - Más parámetros, más capacidad
    - Requiere entrenamiento cuidadoso
    """
    print("🏗️ Creando Modelo 2 (Doble)...")
    
    inputs = keras.Input(shape=(None,), dtype="int32")
    
    # Embeddings usando la capa personalizada
    x = PositionalEmbedding(
        sequence_length=Config.SEQUENCE_LENGTH,
        vocab_size=Config.VOCAB_SIZE,
        embed_dim=Config.EMBED_DIM
    )(inputs)
    
    # PRIMERA capa de atención
    attention1 = layers.MultiHeadAttention(
        num_heads=Config.NUM_HEADS, 
        key_dim=Config.EMBED_DIM
    )(x, x)
    x = layers.LayerNormalization()(x + attention1)
    
    ffn1 = keras.Sequential([
        layers.Dense(Config.LATENT_DIM, activation="relu"),
        layers.Dense(Config.EMBED_DIM)
    ])(x)
    x = layers.LayerNormalization()(x + ffn1)
    
    # SEGUNDA capa de atención (lo que lo hace más poderoso)
    attention2 = layers.MultiHeadAttention(
        num_heads=Config.NUM_HEADS, 
        key_dim=Config.EMBED_DIM
    )(x, x)
    x = layers.LayerNormalization()(x + attention2)
    
    ffn2 = keras.Sequential([
        layers.Dense(Config.LATENT_DIM, activation="relu"),
        layers.Dense(Config.EMBED_DIM)
    ])(x)
    x = layers.LayerNormalization()(x + ffn2)
    
    # Capa de salida
    outputs = layers.Dense(Config.VOCAB_SIZE, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(Config.MODEL2_LR),  # LR más bajo
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print(f"✅ Modelo 2 creado: {model.count_params():,} parámetros")
    return model

# ============================================================================
# PREPARACIÓN DE DATOS (COMPARTIDA)
# ============================================================================

def prepare_data():
    """
    Prepara los datos de IMDb para entrenamiento.
    Se usa para ambos modelos.
    """
    print("📊 Preparando datos de IMDb...")
    
    # Cargar dataset
    (x_train, _), _ = keras.datasets.imdb.load_data()
    word_index = keras.datasets.imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    # Convertir a texto
    def decode_review(text):
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])
    
    text_data = []
    for i in range(min(Config.SAMPLE_SIZE, len(x_train))):
        review_text = decode_review(x_train[i])
        text_data.append(review_text)
    
    # Crear vectorizador
    dataset = tf.data.Dataset.from_tensor_slices(text_data).batch(Config.BATCH_SIZE)
    
    text_vectorization = layers.TextVectorization(
        max_tokens=Config.VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=Config.SEQUENCE_LENGTH,
    )
    text_vectorization.adapt(dataset)
    
    # Preparar secuencias de entrenamiento
    def prepare_sequences(text_batch):
        vectorized = text_vectorization(text_batch)
        x = vectorized[:, :-1]
        y = vectorized[:, 1:]
        return x, y
    
    train_dataset = dataset.map(prepare_sequences)
    
    print("✅ Datos preparados")
    return train_dataset, text_vectorization

def save_vectorizer(text_vectorization):
    """Guardar el vectorizador para usar en la app"""
    # Crear carpeta si no existe
    os.makedirs("saved_models", exist_ok=True)
    
    vectorizer_data = {
        'config': text_vectorization.get_config(),
        'weights': text_vectorization.get_weights(),
        'vocabulary': text_vectorization.get_vocabulary()
    }
    
    with open("saved_models/text_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer_data, f)
    
    print("✅ Vectorizador guardado en saved_models/")

# ============================================================================
# FUNCIONES DE ENTRENAMIENTO
# ============================================================================

def train_model_1(train_dataset, text_vectorization):
    """Entrenar específicamente el Modelo 1"""
    print(f"\n🏋️ ENTRENANDO MODELO 1 ({Config.MODEL1_EPOCHS} épocas)")
    print("=" * 50)
    
    # Crear carpeta si no existe
    os.makedirs("saved_models", exist_ok=True)
    
    # Crear modelo
    model = create_model_1()
    
    # Callbacks para entrenamiento inteligente
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='loss', 
            patience=5,  # Parar si no mejora en 5 épocas
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=3
        )
    ]
    
    # Entrenar
    history = model.fit(
        train_dataset,
        epochs=Config.MODEL1_EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Guardar en la carpeta saved_models
    model.save("saved_models/movie_model_1.keras")
    save_vectorizer(text_vectorization)  # Guardar vectorizador también
    
    final_loss = min(history.history['loss'])
    print(f"✅ Modelo 1 entrenado - Mejor loss: {final_loss:.4f}")
    print("💾 Guardado como: saved_models/movie_model_1.keras")

def train_model_2(train_dataset, text_vectorization):
    """Entrenar específicamente el Modelo 2"""
    print(f"\n🏋️ ENTRENANDO MODELO 2 ({Config.MODEL2_EPOCHS} épocas)")
    print("=" * 50)
    
    # Crear carpeta si no existe
    os.makedirs("saved_models", exist_ok=True)
    
    # Crear modelo
    model = create_model_2()
    
    # Callbacks MÁS RESTRICTIVOS para evitar overfitting
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='loss', 
            patience=3,  # MENOS paciencia
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.7,  # Reducción más suave
            patience=2
        )
    ]
    
    # Entrenar
    history = model.fit(
        train_dataset,
        epochs=Config.MODEL2_EPOCHS,  # MENOS épocas
        callbacks=callbacks,
        verbose=1
    )
    
    # Guardar en la carpeta saved_models
    model.save("saved_models/movie_model_2.keras")
    save_vectorizer(text_vectorization)
    
    final_loss = min(history.history['loss'])
    print(f"✅ Modelo 2 entrenado - Mejor loss: {final_loss:.4f}")
    print("💾 Guardado como: saved_models/movie_model_2.keras")

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal con argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description='Entrenar modelos Transformer')
    parser.add_argument('--modelo', choices=['1', '2', 'all'], required=True,
                       help='Qué modelo entrenar: 1, 2, o all (ambos)')
    
    args = parser.parse_args()
    
    print("🚀 ENTRENAMIENTO DE MODELOS TRANSFORMER")
    print("=" * 50)
    print(f"🎯 Entrenando: Modelo {args.modelo}")
    print(f"📊 Configuración:")
    print(f"   - Vocabulario: {Config.VOCAB_SIZE} tokens")
    print(f"   - Secuencia: {Config.SEQUENCE_LENGTH} tokens")
    print(f"   - Muestras: {Config.SAMPLE_SIZE}")
    
    # Preparar datos (común para ambos)
    train_dataset, text_vectorization = prepare_data()
    
    # Entrenar según selección
    if args.modelo == '1':
        train_model_1(train_dataset, text_vectorization)
    elif args.modelo == '2':
        train_model_2(train_dataset, text_vectorization)
    elif args.modelo == 'all':
        train_model_1(train_dataset, text_vectorization)
        print("\n" + "="*30 + " MODELO 2 " + "="*30)
        train_model_2(train_dataset, text_vectorization)
    
    print("\n🎉 ¡ENTRENAMIENTO COMPLETADO!")
    print("🚀 Ahora puedes usar: streamlit run app.py")

# ============================================================================
# MODO INTERACTIVO (si no se usan argumentos)
# ============================================================================

def interactive_mode():
    """Modo interactivo si se ejecuta sin argumentos"""
    print("🎬 ENTRENADOR DE MODELOS DE RESEÑAS")
    print("=" * 40)
    print("¿Qué modelo quieres entrenar?")
    print("1. Modelo 1 (Simple, estable)")
    print("2. Modelo 2 (Doble, más complejo)")
    print("3. Ambos modelos")
    print("4. Salir")
    
    while True:
        choice = input("\nElige una opción (1-4): ").strip()
        
        if choice in ['1', '2', '3']:
            # Preparar datos
            train_dataset, text_vectorization = prepare_data()
            
            if choice == '1':
                train_model_1(train_dataset, text_vectorization)
            elif choice == '2':
                train_model_2(train_dataset, text_vectorization)
            elif choice == '3':
                train_model_1(train_dataset, text_vectorization)
                train_model_2(train_dataset, text_vectorization)
            
            print("\n✅ ¡Entrenamiento completado!")
            break
            
        elif choice == '4':
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción no válida.")

if __name__ == "__main__":
    import sys
    
    # Si no hay argumentos, usar modo interactivo
    if len(sys.argv) == 1:
        interactive_mode()
    else:
        main()