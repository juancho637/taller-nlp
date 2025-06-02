"""
🤖 MODELO 1 - TRANSFORMER SIMPLE
================================

Modelo 1: Una capa Transformer para generación estable y coherente.

Características:
- ✅ Una sola capa de atención
- ✅ Entrenamiento estable y rápido
- ✅ Menos propenso al overfitting
- ✅ Ideal para principiantes
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers
import os

# ============================================================================
# CONFIGURACIÓN MODELO 1
# ============================================================================

class Model1Config:
    """Configuración específica para el Modelo 1"""
    
    # Identificación
    MODEL_NAME = "1"
    MODEL_DESCRIPTION = "Transformer Simple (Una capa de atención)"
    
    # Arquitectura específica
    NUM_TRANSFORMER_LAYERS = 1      # Solo UNA capa transformer
    DROPOUT_RATE = 0.1              # Dropout bajo para estabilidad
    
    # Parámetros del modelo (heredados de configuración base)
    VOCAB_SIZE = 5000
    SEQUENCE_LENGTH = 60
    EMBED_DIM = 128
    NUM_HEADS = 4
    LATENT_DIM = 256
    
    # Entrenamiento
    EPOCHS = 15                     # Menos épocas, convergencia estable
    LEARNING_RATE = 0.001           # Learning rate estándar
    BATCH_SIZE = 32
    
    # Callbacks
    EARLY_STOPPING_PATIENCE = 5     # Más paciencia para convergencia estable
    REDUCE_LR_PATIENCE = 3
    REDUCE_LR_FACTOR = 0.5
    
    # Objetivos
    TARGET_LOSS = 4.5               # Pérdida objetivo realista
    
    @classmethod
    def print_info(cls):
        """Imprime información del modelo"""
        print("🤖 MODELO 1 - TRANSFORMER SIMPLE")
        print("=" * 45)
        print(f"📝 Descripción: {cls.MODEL_DESCRIPTION}")
        print(f"🏗️ Capas Transformer: {cls.NUM_TRANSFORMER_LAYERS}")
        print(f"🎓 Épocas: {cls.EPOCHS}")
        print(f"📈 Learning Rate: {cls.LEARNING_RATE}")
        print(f"🎯 Pérdida objetivo: {cls.TARGET_LOSS}")
        print(f"💧 Dropout: {cls.DROPOUT_RATE}")
        print()
        print("✨ Fortalezas:")
        print("   - Entrenamiento estable")
        print("   - Convergencia rápida")
        print("   - Menos overfitting")
        print("   - Ideal para principiantes")
        print("   - Resultados predecibles")
        print()
        print("🎯 Mejor para:")
        print("   - Usuarios nuevos en NLP")
        print("   - Aplicaciones que requieren estabilidad")
        print("   - Texto coherente y conservador")

# ============================================================================
# CAPA PERSONALIZADA (COMPARTIDA)
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

# ============================================================================
# ARQUITECTURA DEL MODELO 1
# ============================================================================

def create_model_1():
    """
    Crea el Modelo 1: Transformer Simple con una sola capa de atención.
    
    Arquitectura:
    - Embeddings posicionales
    - 1 capa de Multi-Head Attention
    - 1 red Feed-Forward
    - Layer Normalization
    - Capa de salida
    
    Returns:
        Modelo compilado listo para entrenar
    """
    print("🏗️ Creando Modelo 1 - Transformer Simple...")
    
    config = Model1Config()
    
    # Entrada
    inputs = keras.Input(shape=(None,), dtype="int32")
    
    # Embeddings de tokens y posiciones
    x = PositionalEmbedding(
        sequence_length=config.SEQUENCE_LENGTH,
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBED_DIM
    )(inputs)
    
    # ÚNICA capa de atención múltiple
    attention = layers.MultiHeadAttention(
        num_heads=config.NUM_HEADS, 
        key_dim=config.EMBED_DIM,
        dropout=config.DROPOUT_RATE
    )(x, x)
    
    # Conexión residual + Layer Norm
    x = layers.LayerNormalization()(x + attention)
    
    # Red Feed-Forward
    ffn = keras.Sequential([
        layers.Dense(config.LATENT_DIM, activation="relu"),
        layers.Dropout(config.DROPOUT_RATE),
        layers.Dense(config.EMBED_DIM)
    ])(x)
    
    # Conexión residual + Layer Norm
    x = layers.LayerNormalization()(x + ffn)
    
    # Capa de salida - predicción de siguiente palabra
    outputs = layers.Dense(config.VOCAB_SIZE, activation="softmax")(x)
    
    # Crear y compilar modelo
    model = keras.Model(inputs, outputs, name="transformer_model_1")
    
    model.compile(
        optimizer=keras.optimizers.Adam(config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    total_params = model.count_params()
    print(f"✅ Modelo 1 creado exitosamente")
    print(f"📊 Parámetros totales: {total_params:,}")
    print(f"🎯 Configurado para {config.EPOCHS} épocas con LR={config.LEARNING_RATE}")
    
    return model

# ============================================================================
# ENTRENAMIENTO DEL MODELO 1
# ============================================================================

def train_model_1(train_dataset, text_vectorization, verbose=True):
    """
    Entrena el Modelo 1 con configuración optimizada.
    
    Args:
        train_dataset: Dataset de entrenamiento preparado
        text_vectorization: Vectorizador de texto
        verbose: Si mostrar información detallada
        
    Returns:
        tuple: (modelo_entrenado, historial_entrenamiento)
    """
    config = Model1Config()
    
    if verbose:
        print(f"\n🏋️ ENTRENANDO MODELO 1")
        print("=" * 50)
        config.print_info()
    
    # Crear directorio si no existe
    os.makedirs("saved_models", exist_ok=True)
    
    # Crear modelo
    model = create_model_1()
    
    if verbose:
        print("\n📋 Resumen del modelo:")
        model.summary()
    
    # Configurar callbacks para entrenamiento estable
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1 if verbose else 0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            verbose=1 if verbose else 0,
            min_lr=0.0001
        ),
        keras.callbacks.ModelCheckpoint(
            filepath="saved_models/movie_model_1_best.keras",
            monitor='loss',
            save_best_only=True,
            verbose=1 if verbose else 0
        )
    ]
    
    # Entrenar modelo
    if verbose:
        print(f"\n🚀 Iniciando entrenamiento...")
        print(f"⏱️ Épocas máximas: {config.EPOCHS}")
        print(f"🎯 Pérdida objetivo: {config.TARGET_LOSS}")
    
    history = model.fit(
        train_dataset,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1 if verbose else 2
    )
    
    # Guardar modelo final
    model.save("saved_models/movie_model_1.keras")
    
    # Mostrar resultados
    final_loss = min(history.history['loss'])
    final_accuracy = max(history.history['accuracy'])
    
    if verbose:
        print(f"\n✅ ENTRENAMIENTO MODELO 1 COMPLETADO")
        print("=" * 50)
        print(f"📉 Mejor pérdida: {final_loss:.4f}")
        print(f"📈 Mejor precisión: {final_accuracy:.4f}")
        print(f"🎯 Objetivo alcanzado: {'✅ SÍ' if final_loss <= config.TARGET_LOSS else '❌ NO'}")
        print(f"💾 Modelo guardado en: saved_models/movie_model_1.keras")
        print(f"📊 Épocas ejecutadas: {len(history.history['loss'])}")
    
    return model, history

# ============================================================================
# FUNCIÓN DE UTILIDAD
# ============================================================================

def get_model_1_info():
    """
    Obtiene información completa del Modelo 1.
    
    Returns:
        dict: Información del modelo
    """
    config = Model1Config()
    
    return {
        'name': config.MODEL_NAME,
        'description': config.MODEL_DESCRIPTION,
        'transformer_layers': config.NUM_TRANSFORMER_LAYERS,
        'epochs': config.EPOCHS,
        'learning_rate': config.LEARNING_RATE,
        'target_loss': config.TARGET_LOSS,
        'dropout_rate': config.DROPOUT_RATE,
        'strengths': [
            "Entrenamiento estable",
            "Convergencia rápida",
            "Menos overfitting",
            "Ideal para principiantes",
            "Resultados predecibles"
        ],
        'best_for': [
            "Usuarios nuevos en NLP",
            "Aplicaciones que requieren estabilidad",
            "Texto coherente y conservador"
        ],
        'recommended_temperature': 0.7,
        'recommended_length': 40
    }

# Registrar la capa personalizada
keras.utils.get_custom_objects()['PositionalEmbedding'] = PositionalEmbedding