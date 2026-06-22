import numpy as np
import tensorflow as tf

def resolver_con_deep_learning(matriz_9x9, ruta_modelo="cnn_1m.keras"):
    """
    Toma la matriz leída por OpenCV, la transforma a (1, 81),
    le pide la solución al modelo y la vuelve a transformar en 9x9.
    """
    # 1. Estirar la matriz 9x9 para tener una lista plana de 81 números
    lista_81 = matriz_9x9.flatten()  # Ejemplo: [6, 4, 0, 0, ..., 9, 0]
    
    # 2. Transformaciones necesarias según tu info_modelo.md:
    # Convertimos a array de numpy y le añadimos la dimensión extra -> shape (1, 81)
    entrada_modelo = np.array(lista_81).reshape(1, 81).astype("float32")
    
    # 3. Cargar el modelo y predecir
    modelo_resolver = tf.keras.models.load_model(ruta_modelo)
    prediccion = modelo_resolver.predict(entrada_modelo, verbose=0)
    
    # 4. Procesar la salida del modelo:
    # El modelo devuelve probabilidades para cada una de las 81 celdas (valores del 1 al 9).
    # Usamos np.argmax en el último eje para quedarnos con el número que tiene más probabilidad.
    lista_resuelta = np.argmax(prediccion, axis=-1)
    
    # 5. Volver a darle forma de tablero 9x9 para poder mostrarlo en Streamlit
    matriz_resuelta = lista_resuelta.reshape(9, 9)
    
    return matriz_resuelta