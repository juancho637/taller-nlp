# 🎬 Generador de Reseñas de Películas

Un generador automático de reseñas cinematográficas utilizando modelos Transformer entrenados con el dataset IMDb.

## ✨ Características

- 🤖 **Dos modelos Transformer diferentes** con arquitecturas optimizadas
- 🎛️ **Control de creatividad** mediante temperatura ajustable
- 📏 **Longitud configurable** de las reseñas generadas
- 🎬 **Vocabulario cinematográfico real** de 6000 tokens del dataset IMDb
- 📱 **Interfaz web interactiva** con Streamlit
- ⚡ **Generación en tiempo real** con filtrado inteligente

## 🛠️ Tecnologías

- **TensorFlow 2.19+** - Framework de deep learning
- **Streamlit** - Interfaz web interactiva  
- **Python 3.11+** - Lenguaje de programación
- **Transformer Architecture** - Modelos de atención personalizada

## 🚀 Instalación Rápida

### Opción 1: Con UV (Recomendado)
```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/taller-nlp.git
cd taller-nlp

# Instalar dependencias
uv sync

# Ejecutar aplicación
./run.sh
```

### Opción 2: Con pip tradicional
```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/taller-nlp.git
cd taller-nlp

# Crear entorno virtual
python3.11 -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
streamlit run src/app.py
```

## 🎯 Uso

### Ejecutar la aplicación
```bash
./run.sh
```
La aplicación estará disponible en: http://localhost:8501

### Configuraciones disponibles:
- **Modelo**: Transformer Modelo 1 (preciso) o Modelo 2 (creativo)
- **Temperatura**: 0.1-2.0 (creatividad del texto)
- **Longitud**: 20-200 palabras

### Ejemplos de prompts:
- `"Esta película de acción"`
- `"El drama protagonizado por"`
- `"Una comedia que nos presenta"`

## 🏋️ Entrenamiento de Modelos

### Entrenar ambos modelos:
```bash
cd src
python train.py --modelo ambos --epocas1 40 --epocas2 25
```

### Entrenar solo Modelo 1:
```bash
python train.py --modelo 1 --epocas1 40
```

### Entrenar solo Modelo 2:
```bash
python train.py --modelo 2 --epocas2 20
```

### Opciones avanzadas:
```bash
# Limpiar modelos anteriores
python train.py --modelo 1 --limpiar

# Crear nuevo vectorizador
python train.py --modelo 1 --nuevo-vectorizador

# Configuración personalizada
python train.py --modelo ambos --epocas1 50 --epocas2 15 --limpiar
```

## 📊 Arquitectura del Proyecto

```
src/
├── app.py                  # Interfaz Streamlit principal
├── config.py               # Configuraciones del modelo
├── train.py                # Script de entrenamiento
├── models/
│   └── transformer_models.py  # Arquitecturas Transformer
└── utils/
    ├── data_utils.py       # Utilidades de procesamiento de datos
    └── text_generator.py   # Lógica de generación de texto
```

## 🧠 Modelos

### Transformer Modelo 1
- **Parámetros**: ~7.8M
- **Características**: Una capa transformer, convergencia estable
- **Uso recomendado**: Texto preciso y coherente

### Transformer Modelo 2  
- **Parámetros**: ~12.5M
- **Características**: Dos capas transformer, más expresivo
- **Uso recomendado**: Texto creativo y variado

## ⚙️ Configuración

### Parámetros del modelo (config.py):
```python
SEQUENCE_LENGTH = 80      # Longitud de secuencia
VOCAB_SIZE = 6000         # Tamaño del vocabulario
EMBED_DIM = 256          # Dimensión de embeddings
LATENT_DIM = 1024        # Dimensión latente
NUM_HEADS = 8            # Cabezas de atención
```

### Parámetros de entrenamiento:
```python
BATCH_SIZE = 16          # Tamaño del batch
EPOCHS = 40              # Épocas máximas
SAMPLE_SIZE = 8000       # Muestras del dataset
LEARNING_RATE = 0.0005   # Tasa de aprendizaje
```

## 🎮 Ejemplos de Uso

### Generar reseña de acción:
**Input**: `"Esta película de acción"`  
**Output**: `"Esta película de acción delivers outstanding performances with brilliant cinematography and explosive sequences that keep viewers engaged throughout the entire runtime."`

### Generar reseña de drama:
**Input**: `"El drama protagonizado por"`  
**Output**: `"El drama protagonizado por talented actors explores deep emotional themes with exceptional direction and compelling storytelling that resonates with audiences."`

## 🔧 Solución de Problemas

### Modelos no encontrados:
```bash
# Entrenar modelos desde cero
cd src
python train.py --modelo ambos --limpiar
```

### Error de memoria:
- Reducir `BATCH_SIZE` en `config.py`
- Reducir `SAMPLE_SIZE` para usar menos datos

### Generación incoherente:
- Ajustar temperatura (0.5-0.7 para más coherencia)
- Usar Modelo 1 para mayor precisión

## 📈 Rendimiento

- **Tiempo de entrenamiento**: 25-40 minutos (Mac M1 Pro)
- **Velocidad de generación**: ~2-3 segundos por reseña
- **Precisión**: Loss final ~0.04-4.5 dependiendo del modelo
- **Vocabulario**: 6000 tokens de palabras reales de cine

## 🙏 Reconocimientos

- Dataset IMDb para entrenamiento
- TensorFlow team por el framework
- Streamlit por la interfaz web
- Arquitectura Transformer original de "Attention Is All You Need"