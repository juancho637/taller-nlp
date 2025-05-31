# """
# Configuraciones principales del Generador de Reseñas
# Aquí están todos los parámetros importantes como en tu notebook
# """

# # Parámetros del modelo (basados en tu notebook)
# SEQUENCE_LENGTH = 60
# VOCAB_SIZE = 8000
# # EMBED_DIM = 256
# EMBED_DIM = 128
# LATENT_DIM = 512
# NUM_HEADS = 4

# # Parámetros de entrenamiento
# # BATCH_SIZE = 256
# BATCH_SIZE = 32
# # EPOCHS = 200
# EPOCHS = 10
# SAMPLE_SIZE = 5000   # Solo una muestra del dataset
# LEARNING_RATE = 0.001


# # Parámetros de generación de texto
# GENERATE_LENGTH = 50
# TEMPERATURES = [0.2, 0.5, 0.7, 1.0, 1.5]

# # Rutas de archivo
# DATA_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
# DATA_DIR = "aclImdb"
# MODEL_SAVE_PATH = "saved_models/"

# # Configuración de Streamlit
# DEFAULT_PROMPT = "Esta película"
# DEFAULT_TEMPERATURE = 0.7
# DEFAULT_LENGTH = 100


# """
# Configuración optimizada para entrenamiento rápido (30 minutos)
# """

# # Parámetros del modelo (optimizados para velocidad)
# SEQUENCE_LENGTH = 40  # Reducido de 60 (menos memoria, más rápido)
# VOCAB_SIZE = 5000     # Reducido de 8000 (vocabulario más pequeño)
# EMBED_DIM = 64        # Reducido de 128 (modelo más pequeño)
# LATENT_DIM = 256      # Reducido de 512 (menos cálculos)
# NUM_HEADS = 2         # Reducido de 4 (menos attention heads)

# # Parámetros de entrenamiento (OPTIMIZADO PARA VELOCIDAD)
# BATCH_SIZE = 64       # Aumentado de 32 (menos iteraciones)
# EPOCHS = 20           # ÓPTIMO para 30 minutos
# SAMPLE_SIZE = 3000    # Reducido de 5000 (menos datos, más rápido)
# LEARNING_RATE = 0.002 # Aumentado para convergencia más rápida

# # Parámetros de generación de texto
# GENERATE_LENGTH = 50
# TEMPERATURES = [0.2, 0.5, 0.7, 1.0, 1.5]

# # Rutas de archivo
# DATA_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
# DATA_DIR = "aclImdb"
# MODEL_SAVE_PATH = "saved_models/"

# # Configuración de Streamlit
# DEFAULT_PROMPT = "Esta película"
# DEFAULT_TEMPERATURE = 0.7
# DEFAULT_LENGTH = 50  # Reducido para generación más rápida


"""
Configuración optimizada para MÁXIMA PRECISIÓN Y COHERENCIA
"""

# Parámetros del modelo (OPTIMIZADOS PARA CALIDAD)
SEQUENCE_LENGTH = 80      # AUMENTADO para más contexto
VOCAB_SIZE = 6000         # Balance entre velocidad y vocabulario
EMBED_DIM = 256          # AUMENTADO para mejor representación
LATENT_DIM = 1024        # AUMENTADO para más capacidad
NUM_HEADS = 8            # AUMENTADO para mejor atención

# Parámetros de entrenamiento (PRECISIÓN MÁXIMA)
BATCH_SIZE = 16          # REDUCIDO para mejor gradiente
EPOCHS = 40              # AUMENTADO significativamente
SAMPLE_SIZE = 8000       # AUMENTADO para más datos
LEARNING_RATE = 0.0005   # REDUCIDO para convergencia estable

# Parámetros de generación de texto
GENERATE_LENGTH = 100
TEMPERATURES = [0.2, 0.4, 0.6, 0.8, 1.0]

# Rutas de archivo
DATA_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATA_DIR = "aclImdb"
MODEL_SAVE_PATH = "saved_models/"

# Configuración de Streamlit
DEFAULT_PROMPT = "This movie"
DEFAULT_TEMPERATURE = 0.5  # MÁS CONSERVADOR
DEFAULT_LENGTH = 60