import streamlit as st
from PIL import Image
import traceback
import numpy as np

# Importamos tus modelos
from modelo_yolo.modelo_1 import detectar_y_recortar_sudoku
from modelo_class.modelo_2 import (
    dividir_tablero,
    generar_matriz
)
from modelo_juego.modelo_3 import resolver_sudoku

st.set_page_config(
    page_title="Sudoku Solver",
    page_icon="🧩",
    layout="wide"
)

# PORTADA
st.image(
    "app/sugoku_51.png",
    use_container_width=True
)

st.title("SUDOKU")
st.subheader("Resolución automática de Sudokus mediante Deep Learning")

st.write(
    """
    Sube una imagen que contenga un Sudoku.
    El sistema detectará el tablero, reconocerá los números
    y mostrará la solución.
    """
)

uploaded_file = st.file_uploader(
    "Selecciona una imagen",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Imagen cargada",
        width=350
    )

    if st.button("Resolver Sudoku"):
        try:
            with st.spinner("Procesando imagen..."):

                # =====================
                # GUARDAR IMAGEN USUARIO
                # =====================
                ruta_imagen = "imagen_usuario.png"
                image.save(ruta_imagen)

                # =====================
                # MODELO 1 (YOLO) - Detección de tablero
                # =====================
                detectar_y_recortar_sudoku(
                    imagen_entrada=ruta_imagen,
                    modelo_sudoku="modelo_yolo/best_1.pt",
                    salida="imagen_recortada2.png"
                )

                st.image(
                    "imagen_recortada2.png",
                    caption="Tablero recortado por YOLO",
                    width=400
                )

                # =====================
                # MODELO 2 (OCR) - Extracción de números
                # =====================
                
                dividir_tablero("imagen_recortada2.png") 
                
                
                tablero_detectado = generar_matriz(
                    carpeta_celdas="celdas", 
                    ruta_modelo="modelo_class/modelo2_cnn.keras"
                )

                st.subheader("Matriz detectada por el OCR")
                st.table(tablero_detectado)

                # =====================
                # MODELO 3 (BACKTRACKING) - Solución
                # =====================
                
                solucion = resolver_sudoku(tablero_detectado)

                
                tablero_vis = tablero_detectado.tolist()
                solucion_vis = solucion.tolist()

            st.success("Sudoku resuelto correctamente")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Sudoku detectado")
                st.table(tablero_vis)
            
            with col2:
                st.subheader("Solución")
                st.table(solucion_vis)

        except Exception as e:
            st.error(f"Error durante el proceso: {e}")
            st.code(traceback.format_exc())

else:
    st.info("Selecciona una imagen para comenzar.")