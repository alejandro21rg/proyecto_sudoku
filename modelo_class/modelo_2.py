# modelo_2.py

import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model

def dividir_tablero(imagen_tablero, carpeta_salida="celdas"):
    """
    Divide el tablero 9x9 en 81 imágenes individuales aplicando un 
    margen dinámico para eliminar por completo las líneas gruesas del grid.
    """
    img = cv2.imread(imagen_tablero)
    if img is None:
        raise Exception(f"No se pudo leer la imagen: {imagen_tablero}")

    alto, ancho = img.shape[:2]
    cell_h = alto // 9
    cell_w = ancho // 9

    os.makedirs(carpeta_salida, exist_ok=True)
    contador = 0

    # Margen de seguridad (aprox 14-16% del tamaño de la celda)
    margen_h = max(3, int(cell_h * 0.14))
    margen_w = max(3, int(cell_w * 0.14))

    for fila in range(9):
        for columna in range(9):
            y1 = fila * cell_h
            y2 = (fila + 1) * cell_h
            x1 = columna * cell_w
            x2 = (columna + 1) * cell_w

            # Extraemos la celda aplicando el recorte perimetral protector
            celda = img[y1 + margen_h : y2 - margen_h, x1 + margen_w : x2 - margen_w]
            
            ruta_celda = os.path.join(carpeta_salida, f"celda_{contador}.jpg")
            cv2.imwrite(ruta_celda, celda)
            contador += 1

def cargar_modelo(ruta_modelo="modelo_class/modelo2_cnn.keras"):
    """
    Carga el modelo entrenado Keras de forma segura.
    """
    if not os.path.exists(ruta_modelo):
        alternativa = "modelo2_cnn.keras"
        if os.path.exists(alternativa):
            return load_model(alternativa)
        raise FileNotFoundError(f"No se encontró el archivo del modelo en: {ruta_modelo}")
    return load_model(ruta_modelo)


def predict_celda(img_celda, model):
    """
    Procesa una celda aplicando centrado directo y filtros basados en 
    relaciones de aspecto para evitar distorsiones entre 1, 2 y 7.
    """
    if img_celda is None or img_celda.size == 0:
        return 0

    # --- CORRECCIÓN SEGURA DE CANALES ---
    if len(img_celda.shape) == 3:
        if img_celda.shape[2] == 3:  # BGR estándar
            img_gray = cv2.cvtColor(img_celda, cv2.COLOR_BGR2GRAY)
        elif img_celda.shape[2] == 4:  # BGRA
            img_gray = cv2.cvtColor(img_celda, cv2.COLOR_BGRA2GRAY)
        else:  # Gris con dimensión extra (H, W, 1)
            img_gray = img_celda[:, :, 0].copy()
    else:
        img_gray = img_celda.copy()

    # Recorte perimetral de seguridad para quitar bordes de la cuadrícula
    h_g, w_g = img_gray.shape
    img_gray = img_gray[3:h_g-3, 3:w_g-3] 
    h_c, w_c = img_gray.shape

    # Binarización
    _, img_init_bin = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contornos_padre, _ = cv2.findContours(
        img_init_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    img_final = np.zeros((28, 28), dtype=np.uint8)
    encontrado = False
    w_box, h_box = 0, 0  # <--- Inicializadas siempre a 0 aquí arriba para evitar errores

    if contornos_padre:
        # Filtrar contornos que sean demasiado grandes (ruido de bordes)
        contornos_validos = [c for c in contornos_padre if cv2.contourArea(c) < (w_c * h_c * 0.45)]

        if contornos_validos:
            c_numero = max(contornos_validos, key=cv2.contourArea)
            x, y, w_box, h_box = cv2.boundingRect(c_numero)

            # Si el contorno tiene un tamaño mínimo razonable para ser un número
            if w_box >= 2 and h_box >= 4:
                digito = img_init_bin[y : y + h_box, x : x + w_box]

                # Redimensionar manteniendo aspecto (máximo 18 píxeles estilo MNIST)
                factor = min(18 / h_box, 18 / w_box)
                nuevo_w = max(1, int(w_box * factor))
                nuevo_h = max(1, int(h_box * factor))
                digito_scaled = cv2.resize(digito, (nuevo_w, nuevo_h))

                # Centrado simétrico directo en el lienzo de 28x28 (Método Seguro)
                start_y = (28 - nuevo_h) // 2
                start_x = (28 - nuevo_w) // 2
                img_final[start_y : start_y + nuevo_h, start_x : start_x + nuevo_w] = digito_scaled
                encontrado = True

    # Si la celda no tiene contorno o tiene poquísimos píxeles activos, está vacía
    # Bajado a 14 para asegurar que lea los "1" tipográficos finos sin ignorarlos
    if not encontrado or cv2.countNonZero(img_final) < 14:
        return 0

    # Predicción directa con la CNN
    img_cnn = img_final.astype("float32") / 255.0
    img_cnn = img_cnn.reshape(1, 28, 28, 1)

    pred = model.predict(img_cnn, verbose=0)
    clase_predicha = np.argmax(pred)

    # --- REGLAS DE DESEMPATE GEOMÉTRICO ---
    relacion_aspecto = w_box / float(h_box) if h_box > 0 else 0
    
    # Caso A: Cree que es un 1, pero es ancho -> Es un 7
    if clase_predicha == 1:
        if relacion_aspecto > 0.48:
            clase_predicha = 7

    # Caso B: Cree que es un 7
    elif clase_predicha == 7:
        # 1. Si es extremadamente estrecho, es un 1 lineal que confunde a la CNN
        if relacion_aspecto < 0.38:
            clase_predicha = 1
            
        # 2. Si es muy ancho y tiene peso en la base inferior, es un 2
        elif relacion_aspecto > 0.60:
            mitad_inferior = img_final[18:28, :]
            if cv2.countNonZero(mitad_inferior) > (cv2.countNonZero(img_final) * 0.35):
                clase_predicha = 2

    return int(clase_predicha)


def generar_matriz(carpeta_celdas="celdas", ruta_modelo="modelo_class/modelo2_cnn.keras"):
    """
    Genera la matriz 9x9 a partir de las 81 celdas y exporta los archivos correspondientes.
    """
    modelo = cargar_modelo(ruta_modelo)
    tablero = []

    for i in range(81):
        ruta = f"{carpeta_celdas}/celda_{i}.jpg"
        # Forzamos la lectura en escala de grises tal como en tu script de prueba
        img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        valor = predict_celda(img, modelo)
        tablero.append(valor)

    sudoku_matriz = np.array(tablero).reshape(9, 9)
    
    # --- EXPORTAR LA MATRIZ PARA EL MODELO 3 ---
    np.save("sudoku_detectado.npy", sudoku_matriz)
    np.savetxt("sudoku_detectado.txt", sudoku_matriz, fmt="%d")
    
    return sudoku_matriz

# Bloque de ejecución para pruebas locales
if __name__ == "__main__":
    print("Procesando tablero y generando matriz...")
    # Asegúrate de ajustar las rutas si lo ejecutas directamente de forma local
    try:
        matriz = generar_matriz()
        print("\nMatriz del Sudoku generada con éxito:")
        print(matriz)
        print("\n¡Matriz exportada con éxito para el Modelo 3 (.npy y .txt)!")
    except Exception as e:
        print(f"Nota/Error en ejecución directa: {e}")