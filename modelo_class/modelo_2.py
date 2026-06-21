# modelo_2.py

import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model

def dividir_tablero(imagen_tablero, carpeta_salida="celdas"):
    """
    Divide el tablero 9x9 en 81 imágenes individuales.
    """

    img = cv2.imread(imagen_tablero)

    if img is None:
        raise Exception(
            f"No se pudo leer la imagen: {imagen_tablero}"
        )

    alto, ancho = img.shape[:2]

    cell_h = alto // 9
    cell_w = ancho // 9

    os.makedirs(carpeta_salida, exist_ok=True)

    contador = 0

    margen_h = max(2, cell_h // 12)
    margen_w = max(2, cell_w // 12)

    for fila in range(9):

        for columna in range(9):

            y1 = fila * cell_h
            y2 = (fila + 1) * cell_h

            x1 = columna * cell_w
            x2 = (columna + 1) * cell_w

            celda = img[
                y1 + margen_h:y2 - margen_h,
                x1 + margen_w:x2 - margen_w
            ]

            cv2.imwrite(
                f"{carpeta_salida}/celda_{contador}.jpg",
                celda
            )

            contador += 1

    return contador

def cargar_modelo(ruta_modelo="modelo2_cnn.keras"):
    """
    Carga el modelo CNN entrenado.
    """
    return load_model(ruta_modelo)




def procesar_celda(img):

    img_gray = img.copy()

    if img_gray is None or img_gray.size == 0:
        return np.zeros((28, 28), dtype=np.uint8), False, 0, 0

    _, img_init_bin = cv2.threshold(
        img_gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contornos_padre, _ = cv2.findContours(
        img_init_bin,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    img_final = np.zeros((28, 28), dtype=np.uint8)

    encontrado = False
    w_box = 0
    h_box = 0

    if contornos_padre:

        h_c = img_gray.shape[0]
        w_c = img_gray.shape[1]

        contornos_validos = []

        for c in contornos_padre:

            x, y, w_box_c, h_box_c = cv2.boundingRect(c)

            if (
                x <= 0
                or y <= 0
                or (x + w_box_c) >= w_c
                or (y + h_box_c) >= h_c
            ):
                if cv2.contourArea(c) > (w_c * h_c * 0.15):
                    contornos_validos.append(c)

                continue

            contornos_validos.append(c)

        if contornos_validos:

            c_numero = max(
                contornos_validos,
                key=cv2.contourArea
            )

            x, y, w_box, h_box = cv2.boundingRect(c_numero)

            if w_box >= 2 and h_box >= 5:

                digito = img_init_bin[
                    y:y+h_box,
                    x:x+w_box
                ]

                factor = min(
                    18 / h_box,
                    18 / w_box
                )

                nuevo_w = int(w_box * factor)
                nuevo_h = int(h_box * factor)

                digito_scaled = cv2.resize(
                    digito,
                    (nuevo_w, nuevo_h)
                )

                start_y = (28 - nuevo_h) // 2
                start_x = (28 - nuevo_w) // 2

                img_final[
                    start_y:start_y+nuevo_h,
                    start_x:start_x+nuevo_w
                ] = digito_scaled

                encontrado = True

    return img_final, encontrado, w_box, h_box


def predecir_celda(img, modelo):

    img_final, encontrado, w_box, h_box = procesar_celda(img)

    if not encontrado:
        return 0

    if cv2.countNonZero(img_final) < 6:
        return 0

    img_cnn = img_final.astype("float32") / 255.0
    img_cnn = img_cnn.reshape(1, 28, 28, 1)

    pred = modelo.predict(
        img_cnn,
        verbose=0
    )

    clase_predicha = np.argmax(pred)

    relacion_aspecto = w_box / float(h_box)

    # Desempate 1 ↔ 7
    if (
        clase_predicha == 1
        and relacion_aspecto > 0.43
    ):
        clase_predicha = 7

    # Desempate 5 ↔ 6
    if clase_predicha == 5:

        mitad_inferior = img_final[14:, :]

        if cv2.countNonZero(
            mitad_inferior
        ) > (
            cv2.countNonZero(img_final) * 0.58
        ):
            clase_predicha = 6

    return int(clase_predicha)


def generar_matriz(
    carpeta_celdas="celdas",
    ruta_modelo="modelo2_cnn.keras"
):
    """
    Genera la matriz 9x9 a partir de las 81 celdas.
    """

    modelo = cargar_modelo(ruta_modelo)

    tablero = []

    for i in range(81):

        ruta = f"{carpeta_celdas}/celda_{i}.jpg"

        img = cv2.imread(
            ruta,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            tablero.append(0)
            continue

        numero = predecir_celda(img, modelo)

        tablero.append(numero)

    sudoku_matriz = np.array(tablero).reshape(9, 9)

    return sudoku_matriz


def guardar_matriz(
    matriz,
    archivo_npy="sudoku_detectado.npy",
    archivo_txt="sudoku_detectado.txt"
):
    """
    Guarda la matriz para el Modelo 3.
    """

    np.save(archivo_npy, matriz)

    np.savetxt(
        archivo_txt,
        matriz,
        fmt="%d"
    )