"""
🏋️ ENTRENADOR CENTRAL DE MODELOS
===============================

Script principal para entrenar los modelos Transformer por separado.
Estructura simplificada y clara.

Uso:
    python train.py --model 1      # Solo Modelo 1
    python train.py --model 2      # Solo Modelo 2  
    python train.py --model both   # Ambos modelos
    python train.py --info 1       # Información Modelo 1
    python train.py --info 2       # Información Modelo 2
    python train.py --status       # Estado de modelos guardados
"""

import argparse
import sys
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pickle

# Importar modelos
from model_1 import train_model_1, get_model_1_info, Model1Config
from model_2 import train_model_2, get_model_2_info, Model2Config

# ============================================================================
# FUNCIONES DE DATOS COMPARTIDAS
# ============================================================================

def prepare_imdb_data(sample_size=2000, verbose=True):
    """
    Prepara los datos de IMDb para entrenamiento.
    Función compartida por ambos modelos.
    """
    if verbose:
        print("📊 PREPARANDO DATOS DE IMDB")
        print("=" * 40)
        print(f"📈 Muestras a procesar: {sample_size:,}")
        print(f"🔤 Vocabulario máximo: 5,000 tokens")
        print(f"📏 Secuencia máxima: 60 tokens")
    
    # Cargar dataset de IMDb
    (x_train, _), _ = keras.datasets.imdb.load_data()
    word_index = keras.datasets.imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    # Función para decodificar reseñas
    def decode_review(text):
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])
    
    # Convertir a texto
    text_data = []
    for i in range(min(sample_size, len(x_train))):
        review_text = decode_review(x_train[i])
        text_data.append(review_text)
    
    if verbose:
        print(f"✅ {len(text_data)} reseñas procesadas")
    
    # Crear dataset y vectorizador
    dataset = tf.data.Dataset.from_tensor_slices(text_data).batch(32)
    
    text_vectorization = layers.TextVectorization(
        max_tokens=5000,
        output_mode="int",
        output_sequence_length=60,
    )
    text_vectorization.adapt(dataset)
    
    # Preparar secuencias de entrenamiento
    def prepare_sequences(text_batch):
        vectorized = text_vectorization(text_batch)
        x = vectorized[:, :-1]
        y = vectorized[:, 1:]
        return x, y
    
    train_dataset = dataset.map(prepare_sequences)
    
    if verbose:
        print("✅ Dataset de entrenamiento preparado")
    
    return train_dataset, text_vectorization

def save_vectorizer(text_vectorization, verbose=True):
    """Guarda el vectorizador para usar en la aplicación"""
    os.makedirs("saved_models", exist_ok=True)
    
    vectorizer_data = {
        'config': text_vectorization.get_config(),
        'weights': text_vectorization.get_weights(),
        'vocabulary': text_vectorization.get_vocabulary()
    }
    
    with open("saved_models/text_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer_data, f)
    
    if verbose:
        print("✅ Vectorizador guardado en saved_models/")

# ============================================================================
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ============================================================================

def train_single_model(model_name, verbose=True):
    """
    Entrena un modelo específico.
    
    Args:
        model_name: "1" o "2"
        verbose: Si mostrar información detallada
        
    Returns:
        bool: True si el entrenamiento fue exitoso
    """
    if verbose:
        print(f"🚀 INICIANDO ENTRENAMIENTO DEL MODELO {model_name}")
        print("=" * 60)
    
    try:
        # Preparar datos (común para ambos modelos)
        train_dataset, text_vectorization = prepare_imdb_data(verbose=verbose)
        
        # Entrenar según el modelo seleccionado
        if model_name == "1":
            model, history = train_model_1(train_dataset, text_vectorization, verbose=verbose)
        elif model_name == "2":
            model, history = train_model_2(train_dataset, text_vectorization, verbose=verbose)
        else:
            raise ValueError(f"Modelo '{model_name}' no reconocido")
        
        # Guardar vectorizador
        save_vectorizer(text_vectorization, verbose=verbose)
        
        if verbose:
            print(f"\n🎉 ¡ENTRENAMIENTO DEL MODELO {model_name} COMPLETADO!")
            print("🚀 Ahora puedes usar: streamlit run app.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {str(e)}")
        return False

# ============================================================================
# ENTRENAR AMBOS MODELOS
# ============================================================================

def train_both_models(verbose=True):
    """
    Entrena ambos modelos secuencialmente.
    
    Returns:
        tuple: (éxito_modelo_1, éxito_modelo_2)
    """
    if verbose:
        print("🚀 ENTRENAMIENTO DE AMBOS MODELOS")
        print("=" * 60)
        print("📋 Plan de entrenamiento:")
        print("   1️⃣ Modelo 1 (Transformer Simple)")
        print("   2️⃣ Modelo 2 (Transformer Doble)")
        print()
    
    # Preparar datos una sola vez
    train_dataset, text_vectorization = prepare_imdb_data(verbose=verbose)
    
    # Entrenar Modelo 1
    if verbose:
        print("\n" + "="*20 + " MODELO 1 " + "="*20)
    
    success_1 = False
    try:
        model_1, history_1 = train_model_1(train_dataset, text_vectorization, verbose=verbose)
        success_1 = True
    except Exception as e:
        print(f"❌ Error entrenando Modelo 1: {str(e)}")
    
    # Entrenar Modelo 2
    if verbose:
        print("\n" + "="*20 + " MODELO 2 " + "="*20)
    
    success_2 = False
    try:
        model_2, history_2 = train_model_2(train_dataset, text_vectorization, verbose=verbose)
        success_2 = True
    except Exception as e:
        print(f"❌ Error entrenando Modelo 2: {str(e)}")
    
    # Guardar vectorizador
    save_vectorizer(text_vectorization, verbose=verbose)
    
    # Resumen final
    if verbose:
        print(f"\n🏁 RESUMEN FINAL")
        print("=" * 30)
        print(f"🤖 Modelo 1: {'✅ Éxito' if success_1 else '❌ Falló'}")
        print(f"🤖 Modelo 2: {'✅ Éxito' if success_2 else '❌ Falló'}")
        
        if success_1 or success_2:
            print(f"\n🚀 Ahora puedes usar: streamlit run app.py")
    
    return success_1, success_2

# ============================================================================
# FUNCIONES DE INFORMACIÓN
# ============================================================================

def show_model_info(model_name):
    """Muestra información detallada de un modelo"""
    if model_name == "1":
        Model1Config.print_info()
    elif model_name == "2":
        Model2Config.print_info()
    else:
        print(f"❌ Modelo '{model_name}' no reconocido")

def show_models_status():
    """Muestra el estado de los modelos guardados"""
    print("📋 ESTADO DE MODELOS GUARDADOS")
    print("=" * 35)
    
    models_dir = "saved_models"
    model_1_path = f"{models_dir}/movie_model_1.keras"
    model_2_path = f"{models_dir}/movie_model_2.keras"
    vectorizer_path = f"{models_dir}/text_vectorizer.pkl"
    
    print(f"📁 Directorio: {'✅' if os.path.exists(models_dir) else '❌'} {models_dir}/")
    print(f"🤖 Modelo 1: {'✅' if os.path.exists(model_1_path) else '❌'} movie_model_1.keras")
    print(f"🤖 Modelo 2: {'✅' if os.path.exists(model_2_path) else '❌'} movie_model_2.keras")
    print(f"🔧 Vectorizador: {'✅' if os.path.exists(vectorizer_path) else '❌'} text_vectorizer.pkl")
    
    available = []
    if os.path.exists(model_1_path):
        available.append("1")
    if os.path.exists(model_2_path):
        available.append("2")
    
    if available:
        print(f"\n🎯 Modelos disponibles para usar: {', '.join(available)}")
    else:
        print(f"\n⚠️ No hay modelos entrenados. Ejecuta:")
        print(f"   python train.py --model 1")

def compare_models():
    """Compara ambos modelos"""
    print("⚖️ COMPARACIÓN DE MODELOS")
    print("=" * 40)
    
    print("🤖 MODELO 1 - Simple:")
    print("   - 1 capa Transformer")
    print("   - ~7-8M parámetros")
    print("   - 15 épocas, LR=0.001")
    print("   - Más estable y rápido")
    print("   - Ideal para principiantes")
    
    print("\n🤖 MODELO 2 - Doble:")
    print("   - 2 capas Transformer")
    print("   - ~12-13M parámetros")
    print("   - 20 épocas, LR=0.0005")
    print("   - Más creativo pero complejo")
    print("   - Para usuarios avanzados")
    
    print("\n💡 Recomendación:")
    print("   - Empieza con Modelo 1")
    print("   - Experimenta con Modelo 2 después")

# ============================================================================
# MODO INTERACTIVO
# ============================================================================

def interactive_mode():
    """Modo interactivo si no hay argumentos"""
    print("🎬 ENTRENADOR DE MODELOS DE RESEÑAS")
    print("=" * 40)
    print("¿Qué quieres hacer?")
    print("1. Entrenar Modelo 1 (Simple)")
    print("2. Entrenar Modelo 2 (Doble)")
    print("3. Entrenar ambos modelos")
    print("4. Ver información Modelo 1")
    print("5. Ver información Modelo 2")
    print("6. Ver estado de modelos")
    print("7. Comparar modelos")
    print("8. Salir")
    
    while True:
        choice = input("\nElige una opción (1-8): ").strip()
        
        if choice == '1':
            train_single_model("1")
            break
        elif choice == '2':
            train_single_model("2")
            break
        elif choice == '3':
            train_both_models()
            break
        elif choice == '4':
            show_model_info("1")
        elif choice == '5':
            show_model_info("2")
        elif choice == '6':
            show_models_status()
        elif choice == '7':
            compare_models()
        elif choice == '8':
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción no válida.")

# ============================================================================
# FUNCIÓN PRINCIPAL CON ARGUMENTOS
# ============================================================================

def main():
    """Función principal con argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description='Entrenador de Modelos Transformer')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', choices=['1', '2', 'both'], 
                      help='Entrenar modelo específico: 1, 2, o both')
    group.add_argument('--info', choices=['1', '2'],
                      help='Mostrar información del modelo')
    group.add_argument('--status', action='store_true',
                      help='Mostrar estado de modelos guardados')
    group.add_argument('--compare', action='store_true',
                      help='Comparar ambos modelos')
    
    args = parser.parse_args()
    
    # Ejecutar según argumentos
    if args.model:
        if args.model == 'both':
            train_both_models()
        else:
            train_single_model(args.model)
    
    elif args.info:
        show_model_info(args.info)
    
    elif args.status:
        show_models_status()
    
    elif args.compare:
        compare_models()

# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    # Si no hay argumentos, usar modo interactivo
    if len(sys.argv) == 1:
        interactive_mode()
    else:
        main()