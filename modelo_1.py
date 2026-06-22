# MODELO 1 - DETECCIÓN Y RECORTE DEL SUDOKU

import cv2
from ultralytics import YOLO


def detectar_y_recortar_sudoku(
    imagen_entrada,
    modelo_sudoku="best_1.pt",
    salida="imagen_recortada2.png"
):
    """
    Detecta el sudoku en una imagen y devuelve
    un recorte limpio listo para el Modelo 2.
    """

    # Cargar modelo YOLO
    model = YOLO(modelo_sudoku)
 
    # PRIMERA DETECCIÓN
  
    results = model(imagen_entrada)

    resultado = results[0]

    if len(resultado.boxes) == 0:
        raise Exception("No se ha detectado ningún sudoku.")

    x1, y1, x2, y2 = map(int, resultado.boxes.xyxy[0])

    sudoku_recortado = resultado.orig_img[y1:y2, x1:x2]
   
    # SEGUNDA DETECCIÓN
    
    results2 = model(sudoku_recortado)

    resultado2 = results2[0]

    if len(resultado2.boxes) > 0:

        x1, y1, x2, y2 = map(int, resultado2.boxes.xyxy[0])

        sudoku_recortado = sudoku_recortado[y1:y2, x1:x2]

    # Guardar resultado final
    cv2.imwrite(salida, sudoku_recortado)

    print(f"Sudoku guardado en: {salida}")

    return sudoku_recortado


# EJEMPLO DE USO
if __name__ == "__main__":

    detectar_y_recortar_sudoku(
        imagen_entrada="sudoku_online.png",
        modelo_sudoku="best_1.pt",
        salida="imagen_recortada2.png"
    )