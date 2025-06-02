"""
🤖 MODELO 2 - TRANSFORMER DOBLE
===============================

Modelo 2: Dos capas Transformer para generación más creativa y expresiva.

Características:
- ✅ Dos capas de atención
- ✅ Mayor capacidad expresiva
- ✅ Más creativo en la generación
- ✅ Ideal para usuarios avanzados
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers
import os

# ============================================================================
# CONFIGURACIÓN MODELO 2
# ============================================================================

class Model2Config:
    """Configuración específica para el Modelo 2"""
    
    # Identificación
    MODEL_NAME = "2"
    MODEL_DESCRIPTION = "Transformer Doble (Dos capas de atención)"
    
    # Arquitectura específica
    NUM_TRANSFORMER_LAYERS = 2      # DOS capas transformer
    DROPOUT_RATE = 0.2              # Dropout más alto para regularización
    
    # Parámetros del modelo (heredados de configuración base)
    VOCAB_SIZE = 5000
    SEQUENCE_LENGTH = 60
    EMBED_DIM = 128
    NUM_HEADS = 4
    LATENT_DIM = 256
    
    # Entrenamiento (más cuidadoso para evitar overfitting)
    EPOCHS = 20                     # Más épocas pero con early stopping agresivo
    LEARNING_RATE = 0.0005          # Learning rate más bajo para estabilidad
    BATCH_SIZE = 32
    
    # Callbacks más agresivos
    EARLY_STOPPING_PATIENCE = 3     # Menos paciencia para evitar overfitting
    REDUCE_LR_PATIENCE = 2          # Reducir LR más rápido
    REDUCE_LR_FACTOR = 0.7          # Reducción más suave
    
    # Regularización adicional
    WEIGHT_DECAY = 0.0001           # Regularización L2
    GRADIENT_CLIP_NORM = 1.0        # Gradient clipping
    
    # Objetivos
    TARGET_LOSS = 4.0               # Pérdida objetivo más ambiciosa
    
    @classmethod
    def print_info(cls):
        """Imprime información del modelo"""
        print("🤖 MODELO 2 - TRANSFORMER DOBLE")
        print("=" * 45)
        print(f"📝 Descripción: {cls.MODEL_DESCRIPTION}")
        print(f"🏗️ Capas Transformer: {cls.NUM_TRANSFORMER_LAYERS}")
        print(f"🎓 Épocas: {cls.EPOCHS}")
        print(f"📈 Learning Rate: {cls.LEARNING_RATE}")
        print(f"🎯 Pérdida objetivo: {cls.TARGET_LOSS}")
        print(f"💧 Dropout: {cls.DROPOUT_RATE}")
        print(f"⚖️ Weight Decay: {cls.WEIGHT_DECAY}")
        print()
        print("✨ Fortalezas:")
        print("   - Mayor capacidad expresiva")
        print("   - Generación más creativa")
        print("   - Mejor comprensión del contexto")
        print("   - Más variedad en el texto")
        print("   - Ideal para contenido original")
        print()
        print("⚠️ Desafíos:")
        print("   - Más propenso al overfitting")
        print("   - Requiere ajuste cuidadoso")
        print("   - Entrenamiento más lento")
        print()
        print("🎯 Mejor para:")
        print("   - Usuarios avanzados en NLP")
        print("   - Aplicaciones creativas")
        print("   - Texto expresivo y variado")

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
# ARQUITECTURA DEL MODELO 2
# ============================================================================

def create_model_2():
    """
    Crea el Modelo 2: Transformer Doble con dos capas de atención.
    
    Arquitectura:
    - Embeddings posicionales
    - 1ra capa de Multi-Head Attention + FFN
    - 2da capa de Multi-Head Attention + FFN
    - Layer Normalization en cada bloque
    - Capa de salida
    
    Returns:
        Modelo compilado listo para entrenar
    """
    print("🏗️ Creando Modelo 2 - Transformer Doble...")
    
    config = Model2Config()
    
    # Entrada
    inputs = keras.Input(shape=(None,), dtype="int32")
    
    # Embeddings de tokens y posiciones
    x = PositionalEmbedding(
        sequence_length=config.SEQUENCE_LENGTH,
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBED_DIM
    )(inputs)
    
    # PRIMERA capa Transformer
    # Multi-Head Attention 1
    attention1 = layers.MultiHeadAttention(
        num_heads=config.NUM_HEADS,
        key_dim=config.EMBED_DIM,
        dropout=config.DROPOUT_RATE
    )(x, x)
    
    # Conexión residual + Layer Norm
    x = layers.LayerNormalization()(x + attention1)
    
    # Feed-Forward Network 1
    ffn1 = keras.Sequential([
        layers.Dense(config.LATENT_DIM, activation="relu"),
        layers.Dropout(config.DROPOUT_RATE),
        layers.Dense(config.EMBED_DIM)
    ])(x)
    
    # Conexión residual + Layer Norm
    x = layers.LayerNormalization()(x + ffn1)
    
    # SEGUNDA capa Transformer (lo que lo hace más poderoso)
    # Multi-Head Attention 2
    attention2 = layers.MultiHeadAttention(
        num_heads=config.NUM_HEADS,
        key_dim=config.EMBED_DIM,
        dropout=config.DROPOUT_RATE
    )(x, x)
    
    # Conexión residual + Layer Norm
    x = layers.LayerNormalization()(x + attention2)
    
    # Feed-Forward Network 2
    ffn2 = keras.Sequential([
        layers.Dense(config.LATENT_DIM, activation="relu"),
        layers.Dropout(config.DROPOUT_RATE),
        layers.Dense(config.EMBED_DIM)
    ])(x)
    
    # Conexión residual + Layer Norm
    x = layers.LayerNormalization()(x + ffn2)
    
    # Capa de salida - predicción de siguiente palabra
    outputs = layers.Dense(config.VOCAB_SIZE, activation="softmax")(x)
    
    # Crear y compilar modelo
    model = keras.Model(inputs, outputs, name="transformer_model_2")
    
    # Compilar con regularización adicional
    optimizer = keras.optimizers.Adam(
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY  # Regularización L2
    )
    
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    total_params = model.count_params()
    print(f"✅ Modelo 2 creado exitosamente")
    print(f"📊 Parámetros totales: {total_params:,}")
    print(f"🎯 Configurado para {config.EPOCHS} épocas con LR={config.LEARNING_RATE}")
    print(f"🛡️ Regularización: Dropout={config.DROPOUT_RATE}, Weight Decay={config.WEIGHT_DECAY}")
    
    return model

# ============================================================================
# ENTRENAMIENTO DEL MODELO 2
# ============================================================================

def train_model_2(train_dataset, text_vectorization, verbose=True):
    """
    Entrena el Modelo 2 con configuración cuidadosa para evitar overfitting.
    
    Args:
        train_dataset: Dataset de entrenamiento preparado
        text_vectorization: Vectorizador de texto
        verbose: Si mostrar información detallada
        
    Returns:
        tuple: (modelo_entrenado, historial_entrenamiento)
    """
    config = Model2Config()
    
    if verbose:
        print(f"\n🏋️ ENTRENANDO MODELO 2")
        print("=" * 50)
        config.print_info()
    
    # Crear directorio si no existe
    os.makedirs("saved_models", exist_ok=True)
    
    # Crear modelo
    model = create_model_2()
    
    if verbose:
        print("\n📋 Resumen del modelo:")
        model.summary()
    
    # Configurar callbacks MÁS RESTRICTIVOS para evitar overfitting
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=config.EARLY_STOPPING_PATIENCE,  # MENOS paciencia
            restore_best_weights=True,
            verbose=1 if verbose else 0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=config.REDUCE_LR_FACTOR,  # Reducción más suave
            patience=config.REDUCE_LR_PATIENCE,  # Más rápido
            verbose=1 if verbose else 0,
            min_lr=0.00001
        ),
        keras.callbacks.ModelCheckpoint(
            filepath="saved_models/movie_model_2_best.keras",
            monitor='loss',
            save_best_only=True,
            verbose=1 if verbose else 0
        )
    ]
    
    # Agregar gradient clipping si es necesario
    if hasattr(model.optimizer, 'clipnorm'):
        model.optimizer.clipnorm = config.GRADIENT_CLIP_NORM
    
    # Entrenar modelo
    if verbose:
        print(f"\n🚀 Iniciando entrenamiento...")
        print(f"⏱️ Épocas máximas: {config.EPOCHS}")
        print(f"🎯 Pérdida objetivo: {config.TARGET_LOSS}")
        print(f"⚠️ Early stopping agresivo: {config.EARLY_STOPPING_PATIENCE} épocas")
    
    history = model.fit(
        train_dataset,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1 if verbose else 2
    )
    
    # Guardar modelo final
    model.save("saved_models/movie_model_2.keras")
    
    # Mostrar resultados
    final_loss = min(history.history['loss'])
    final_accuracy = max(history.history['accuracy'])
    
    if verbose:
        print(f"\n✅ ENTRENAMIENTO MODELO 2 COMPLETADO")
        print("=" * 50)
        print(f"📉 Mejor pérdida: {final_loss:.4f}")
        print(f"📈 Mejor precisión: {final_accuracy:.4f}")
        print(f"🎯 Objetivo alcanzado: {'✅ SÍ' if final_loss <= config.TARGET_LOSS else '❌ NO'}")
        print(f"💾 Modelo guardado en: saved_models/movie_model_2.keras")
        print(f"📊 Épocas ejecutadas: {len(history.history['loss'])}")
        
        if final_loss > config.TARGET_LOSS:
            print(f"\n💡 Sugerencias para mejorar:")
            print(f"   - Reducir learning rate a 0.0003")
            print(f"   - Aumentar regularización (dropout a 0.3)")
            print(f"   - Usar más datos de entrenamiento")
    
    return model, history

# ============================================================================
# FUNCIÓN DE UTILIDAD
# ============================================================================

def get_model_2_info():
    """
    Obtiene información completa del Modelo 2.
    
    Returns:
        dict: Información del modelo
    """
    config = Model2Config()
    
    return {
        'name': config.MODEL_NAME,
        'description': config.MODEL_DESCRIPTION,
        'transformer_layers': config.NUM_TRANSFORMER_LAYERS,
        'epochs': config.EPOCHS,
        'learning_rate': config.LEARNING_RATE,
        'target_loss': config.TARGET_LOSS,
        'dropout_rate': config.DROPOUT_RATE,
        'weight_decay': config.WEIGHT_DECAY,
        'strengths': [
            "Mayor capacidad expresiva",
            "Generación más creativa",
            "Mejor comprensión del contexto",
            "Más variedad en el texto",
            "Ideal para contenido original"
        ],
        'challenges': [
            "Más propenso al overfitting",
            "Requiere ajuste cuidadoso",
            "Entrenamiento más lento"
        ],
        'best_for': [
            "Usuarios avanzados en NLP",
            "Aplicaciones creativas",
            "Texto expresivo y variado"
        ],
        'recommended_temperature': 0.9,
        'recommended_length': 50
    }

def compare_with_model_1():
    """
    Compara el Modelo 2 con el Modelo 1.
    
    Returns:
        dict: Comparación detallada
    """
    return {
        'complexity': {
            'model_1': '1 capa transformer',
            'model_2': '2 capas transformer',
            'difference': 'Modelo 2 es ~60% más complejo'
        },
        'training': {
            'model_1': 'Más estable, convergencia suave',
            'model_2': 'Más variable, requiere monitoreo',
            'recommendation': 'Usar Modelo 1 primero, luego experimentar con Modelo 2'
        },
        'output_quality': {
            'model_1': 'Coherente y predecible',
            'model_2': 'Más creativo pero menos predecible',
            'use_case': 'Modelo 1 para uso general, Modelo 2 para creatividad'
        }
    }

# Registrar la capa personalizada
keras.utils.get_custom_objects()['PositionalEmbedding'] = PositionalEmbedding