"""
Modelos Transformer para el Generador de Reseñas  
Contiene las clases PositionalEmbedding y TransformerDecoder de tu notebook
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers
import config

class PositionalEmbedding(layers.Layer):
    """
    Capa de embedding posicional (exactamente como en tu notebook)
    """
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = keras.ops.shape(inputs)[-1]
        positions = keras.ops.arange(start=0, stop=length, dtype="int32")
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        # Usar keras.ops en lugar de tf.math para compatibilidad
        return keras.ops.not_equal(inputs, 0)

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config


class TransformerDecoder(layers.Layer):
    """
    Decoder Transformer (exactamente como en tu notebook)
    """
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
          num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(
          num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([
            layers.Dense(dense_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def get_config(self):
        config = super(TransformerDecoder, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config

    def get_causal_attention_mask(self, inputs):
        input_shape = keras.ops.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = keras.ops.arange(sequence_length)[:, None]
        j = keras.ops.arange(sequence_length)
        mask = keras.ops.cast(i >= j, dtype="int32")
        mask = keras.ops.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = keras.ops.concatenate([
            keras.ops.expand_dims(batch_size, -1),
            keras.ops.array([1, 1], dtype="int32")
        ], axis=0)
        return keras.ops.tile(mask, mult)

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = keras.ops.cast(mask[:, None, :], dtype="int32")
            padding_mask = keras.ops.minimum(padding_mask, causal_mask)
        else:
            padding_mask = mask
        
        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs,
            attention_mask=causal_mask)
        attention_output_1 = self.layernorm_1(inputs + attention_output_1)
        
        attention_output_2 = self.attention_2(
            query=attention_output_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        attention_output_2 = self.layernorm_2(
            attention_output_1 + attention_output_2)
        
        proj_output = self.dense_proj(attention_output_2)
        return self.layernorm_3(attention_output_2 + proj_output)


def create_transformer_model_1():
    """
    Modelo Transformer 1 (igual que en tu notebook)
    """
    print("🏗️ Creando Transformer Modelo 1...")
    
    inputs = keras.Input(shape=(None,), dtype="int64")
    
    # Embedding posicional
    x = PositionalEmbedding(
        config.SEQUENCE_LENGTH, 
        config.VOCAB_SIZE, 
        config.EMBED_DIM
    )(inputs)
    
    # Transformer decoder
    x = TransformerDecoder(
        config.EMBED_DIM, 
        config.LATENT_DIM, 
        config.NUM_HEADS
    )(x, x)
    
    # Capa de salida
    outputs = layers.Dense(config.VOCAB_SIZE, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop")
    
    print("✅ Transformer Modelo 1 creado")
    return model


def create_transformer_model_2():
    """
    Modelo Transformer 2 (variación con más capas)
    """
    print("🏗️ Creando Transformer Modelo 2...")
    
    inputs = keras.Input(shape=(None,), dtype="int64")
    
    # Embedding posicional (igual)
    x = PositionalEmbedding(
        config.SEQUENCE_LENGTH, 
        config.VOCAB_SIZE, 
        config.EMBED_DIM
    )(inputs)
    
    # Dos capas Transformer decoder (diferencia principal)
    x = TransformerDecoder(
        config.EMBED_DIM, 
        config.LATENT_DIM, 
        config.NUM_HEADS
    )(x, x)
    
    x = TransformerDecoder(
        config.EMBED_DIM, 
        config.LATENT_DIM, 
        config.NUM_HEADS
    )(x, x)
    
    # Capa de salida
    outputs = layers.Dense(config.VOCAB_SIZE, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop")
    
    print("✅ Transformer Modelo 2 creado")
    return model