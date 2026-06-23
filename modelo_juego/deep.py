import numpy as np
from tensorflow.keras.models import load_model

def resolver_sudoku_dl(matriz_9x9, ruta_modelo="../modelos/cnn_1m.keras"):
    """
    Recibe una matriz de 9x9 (NumPy array), la aplana,
    la prepara para el modelo de Deep Learning y predice la solución.
    """
    # 1. Convertir y aplanar la matriz de 9x9 a una lista/array de 81 números
    lista_81 = matriz_9x9.flatten()
    
    # 2. Añadirle una dimensión simple para cumplir con el shape (1, 81)
    entrada_modelo = np.expand_dims(lista_81, axis=0)  # Shape resultante: (1, 81)
    
    # 3. Cargar el modelo del profesor y predecir
    modelo = load_model(ruta_modelo)
    prediccion = modelo.predict(entrada_modelo)
    
    # Nota: Dependiendo de cómo devuelva la salida el modelo del profesor:
    # Si devuelve directamente los 81 números resueltos con shape (1, 81) o (81,):
    solucion_plana = np.argmax(prediccion, axis=-1) if len(prediccion.shape) > 2 else prediccion
    
    # Si la salida ya es directamente el array de 81 números corregidos, lo reformateamos a 9x9:
    solucion_9x9 = solucion_plana.reshape(9, 9)
    
    return solucion_9x9