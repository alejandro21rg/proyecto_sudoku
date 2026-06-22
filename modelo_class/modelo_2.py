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

    # 🚨 INCREMENTAMOS EL MARGEN DE SEGURIDAD (aprox 14-16% del tamaño de la celda)
    # Esto recorta los bordes negros conflictivos que la CNN confunde con números 1
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
    Procesa una celda aplicando centrado por momentos geométricos 
    para evitar distorsiones entre 1, 5, 6 y 7.
    """
    if img_celda is None or img_celda.size == 0:
        return 0

    if len(img_celda.shape) == 3:
        img_gray = cv2.cvtColor(img_celda[:, :, :3], cv2.COLOR_BGR2GRAY)
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

    if contornos_padre:
        # Filtrar contornos que sean demasiado grandes (ruido de bordes)
        contornos_validos = [c for c in contornos_padre if cv2.contourArea(c) < (w_c * h_c * 0.45)]

        if contornos_validos:
            c_numero = max(contornos_validos, key=cv2.contourArea)
            x, y, w_box, h_box = cv2.boundingRect(c_numero)

            # VALIDACIÓN CRÍTICA: Si el número es extremadamente pequeño, es ruido de fondo
            if h_box < (h_c * 0.20) or w_box < (w_c * 0.10):
                return 0

            if w_box >= 2 and h_box >= 4:
                digito = img_init_bin[y : y + h_box, x : x + w_box]

                # Redimensionar manteniendo aspecto (máximo 18 píxeles)
                factor = min(18 / h_box, 18 / w_box)
                nuevo_w = max(1, int(w_box * factor))
                nuevo_h = max(1, int(h_box * factor))
                digito_scaled = cv2.resize(digito, (nuevo_w, nuevo_h))

                # --- CENTRADO INTELIGENTE POR CENTRO DE MASA ---
                # Creamos un lienzo temporal
                temp_canvas = np.zeros((28, 28), dtype=np.uint8)
                start_y = (28 - nuevo_h) // 2
                start_x = (28 - nuevo_w) // 2
                temp_canvas[start_y : start_y + nuevo_h, start_x : start_x + nuevo_w] = digito_scaled

                # Calculamos los momentos para ajustar al centro de masa real
                M = cv2.moments(temp_canvas)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Desplazamos el dígito para que el centro de masa esté exactamente en (14, 14)
                    shift_x = 14 - cx
                    shift_y = 14 - cy
                    T = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                    img_final = cv2.warpAffine(temp_canvas, T, (28, 28))
                else:
                    img_final = temp_canvas
                
                encontrado = True

    # Si la celda apenas tiene píxeles activos, está vacía
    if not encontrado or cv2.countNonZero(img_final) < 8:
        return 0

    # 3. Predicción directa con la CNN (Sin parches "if" manuales)
    img_cnn = img_final.astype("float32") / 255.0
    img_cnn = img_cnn.reshape(1, 28, 28, 1)

    pred = model.predict(img_cnn, verbose=0)
    clase_predicha = np.argmax(pred)

    return int(clase_predicha)


def generar_matriz(carpeta_celdas="celdas", ruta_modelo="modelo_class/modelo2_cnn.keras"):
    """
    Genera la matriz 9x9 a partir de las 81 celdas.
    """
    modelo = cargar_modelo(ruta_modelo)
    tablero = []

    for i in range(81):
        ruta = f"{carpeta_celdas}/celda_{i}.jpg"
        img = cv2.imread(ruta)
        valor = predict_celda(img, modelo)
        tablero.append(valor)

    sudoku_matriz = np.array(tablero).reshape(9, 9)
    return sudoku_matriz