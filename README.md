# 🎬 Generador de Reseñas de Películas con Transformers

Un generador automático de reseñas cinematográficas utilizando **dos modelos Transformer** con arquitecturas diferenciadas, entrenados con el dataset IMDb.

## ✨ Características Principales

- 🤖 **Dos modelos Transformer independientes** con configuraciones optimizadas
- 🏗️ **Arquitectura modular** - cada modelo en su propio archivo
- 📊 **Información dinámica** - métricas reales extraídas del modelo cargado
- 🎛️ **Control de creatividad** mediante temperatura ajustable
- 📏 **Longitud configurable** de las reseñas generadas
- 📱 **Interfaz web simplificada** con Streamlit
- ⚡ **Entrenamiento por separado** con script centralizado

## 🏗️ Estructura del Proyecto

```
src/
├── train.py        # 🚀 Script central de entrenamiento
├── model_1.py      # 🤖 Modelo 1: Transformer Simple (1 capa)
├── model_2.py      # 🤖 Modelo 2: Transformer Doble (2 capas)
└── app.py          # 📱 Interfaz web con Streamlit
```

### 📋 Descripción de Archivos

| Archivo | Propósito | Contenido |
|---------|-----------|-----------|
| `train.py` | Script principal | Coordinación de entrenamiento, preparación de datos |
| `model_1.py` | Transformer Simple | Configuración y arquitectura de 1 capa |
| `model_2.py` | Transformer Doble | Configuración y arquitectura de 2 capas |
| `app.py` | Interfaz web | Carga de modelos, summary real, generación |

## 🤖 Modelos Implementados

### 🎯 Modelo 1 - Transformer Simple
- **Capas:** 1 capa Transformer
- **Parámetros:** ~7.8M 
- **Épocas:** 15
- **Learning Rate:** 0.001
- **Fortalezas:** Estable, rápido, ideal para principiantes
- **Tiempo entrenamiento:** 10-15 minutos

### 🎯 Modelo 2 - Transformer Doble  
- **Capas:** 2 capas Transformer
- **Parámetros:** ~12.5M
- **Épocas:** 20 
- **Learning Rate:** 0.0005
- **Fortalezas:** Más expresivo, creativo, para usuarios avanzados
- **Tiempo entrenamiento:** 15-25 minutos

### 🔧 Configuración Compartida
```python
VOCAB_SIZE = 5000           # Vocabulario de palabras
SEQUENCE_LENGTH = 60        # Longitud máxima de secuencia
EMBED_DIM = 128            # Dimensión de embeddings
NUM_HEADS = 4              # Cabezas de atención
SAMPLE_SIZE = 2000         # Muestras de entrenamiento IMDb
```

## 🚀 Instalación y Uso

### 📦 Instalación
```bash
# Clonar repositorio
git clone <tu-repositorio>
cd generador-resenas-transformer

# Instalar dependencias
pip install tensorflow streamlit numpy matplotlib
```

### 🏋️ Entrenamiento de Modelos

#### Comandos Principales
```bash
# Entrenar solo Modelo 1 (recomendado para empezar)
python src/train.py --model 1

# Entrenar solo Modelo 2 (más avanzado)
python src/train.py --model 2

# Entrenar ambos modelos secuencialmente
python src/train.py --model both
```

#### Comandos de Información
```bash
# Ver configuración del Modelo 1
python src/train.py --info 1

# Ver configuración del Modelo 2
python src/train.py --info 2

# Ver estado de modelos guardados
python src/train.py --status

# Comparar ambos modelos
python src/train.py --compare
```

#### Modo Interactivo
```bash
# Sin argumentos para menú interactivo
python src/train.py
```

### 📱 Ejecutar la Aplicación Web
```bash
# Iniciar interfaz web (requiere modelos entrenados)
streamlit run src/app.py
```

La aplicación estará disponible en: **http://localhost:8501**

## 🎮 Uso de la Aplicación

### 🤖 Selección de Modelo
- Escoge entre **Modelo 1** (simple) o **Modelo 2** (doble)
- La aplicación carga automáticamente el modelo desde `saved_models/`

### 📊 Ver Detalles del Modelo
- Botón **"📋 Mostrar Detalles del Modelo X"**
- Información extraída dinámicamente:
  - Parámetros totales y entrenables
  - Arquitectura completa con `model.summary()`
  - No hay información hardcodeada

### ✍️ Generación de Reseñas
1. **Escribe un prompt inicial:** "This movie is"
2. **Ajusta la creatividad:** 0.1 (conservador) - 2.0 (muy creativo)
3. **Define la longitud:** 10-100 palabras
4. **Genera:** Obtén tu reseña personalizada

### 📊 Configuraciones Recomendadas

| Modelo | Temperatura | Longitud | Mejor para |
|--------|-------------|----------|------------|
| **Modelo 1** | 0.5 - 0.8 | 30-50 palabras | Texto coherente y predecible |
| **Modelo 2** | 0.7 - 1.2 | 40-80 palabras | Contenido creativo y variado |

## 🔧 Configuraciones Técnicas

### 🏗️ Arquitectura del Modelo 1
```python
# src/model_1.py
class Model1Config:
    NUM_TRANSFORMER_LAYERS = 1      # Una capa
    DROPOUT_RATE = 0.1              # Dropout bajo
    EPOCHS = 15                     # Entrenamiento estable
    EARLY_STOPPING_PATIENCE = 5     # Más paciencia
```

### 🏗️ Arquitectura del Modelo 2
```python
# src/model_2.py  
class Model2Config:
    NUM_TRANSFORMER_LAYERS = 2      # Dos capas
    DROPOUT_RATE = 0.2              # Más regularización
    EPOCHS = 20                     # Más épocas
    EARLY_STOPPING_PATIENCE = 3     # Menos paciencia (evitar overfitting)
```

## 📊 Información Técnica

### 🎯 Dataset
- **Fuente:** IMDb Movie Reviews
- **Muestras:** 2,000 reseñas de entrenamiento
- **Vocabulario:** 5,000 palabras más frecuentes
- **Preprocesamiento:** Tokenización con TextVectorization de Keras

### 🧠 Arquitectura Transformer
- **Embeddings posicionales:** Suma de token + posición
- **Multi-Head Attention:** 4 cabezas de atención
- **Feed-Forward:** Dimensión latente de 256
- **Layer Normalization:** Aplicada después de cada bloque
- **Dropout:** Regularización para evitar overfitting

### 💾 Archivos Generados
```
saved_models/
├── movie_model_1.keras         # Modelo 1 entrenado
├── movie_model_2.keras         # Modelo 2 entrenado
└── text_vectorizer.pkl         # Vectorizador compartido
```

## 🔍 Solución de Problemas

### ❌ Error: "No se encontró el modelo"
```bash
# Entrenar el modelo faltante
python src/train.py --model 1  # o --model 2
```

### ❌ Error: "No se encontró text_vectorizer.pkl"
```bash
# El vectorizador se crea automáticamente al entrenar cualquier modelo
python src/train.py --model 1
```

### 🐌 Entrenamiento muy lento
- Reduce `SAMPLE_SIZE` en las configuraciones
- Usa GPU si está disponible
- Considera entrenar solo Modelo 1 primero

### 🤖 Generación incoherente
- Reduce la temperatura (0.3-0.6)
- Usa Modelo 1 para mayor estabilidad
- Verifica que el modelo esté bien entrenado

## 📈 Rendimiento Esperado

| Métrica | Modelo 1 | Modelo 2 |
|---------|----------|----------|
| **Parámetros** | ~7.8M | ~12.5M |
| **Tiempo entrenamiento** | 10-15 min | 15-25 min |
| **Loss final** | ~4.5 | ~4.0 |
| **Estabilidad** | Alta | Media |
| **Creatividad** | Media | Alta |

## 🎯 Ejemplos de Uso

### 📝 Ejemplo 1 - Modelo 1 (Conservador)
**Input:** `"This movie is"`  
**Temperatura:** 0.6  
**Output:** `"This movie is a compelling drama that showcases excellent performances and delivers a satisfying emotional journey."`

### 📝 Ejemplo 2 - Modelo 2 (Creativo)
**Input:** `"The film presents"`  
**Temperatura:** 1.0  
**Output:** `"The film presents an innovative narrative structure that challenges conventional storytelling while maintaining audience engagement through unexpected plot developments."`