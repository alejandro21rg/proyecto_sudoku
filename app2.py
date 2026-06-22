import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import traceback
import numpy as np
import os 
import shutil

# Importamos tus modelos
from modelo_yolo.modelo_1 import detectar_y_recortar_sudoku
from modelo_class.modelo_2 import (
    dividir_tablero,
    generar_matriz
)
from modelo_juego.modelo_3 import resolver_sudoku

def mostrar_sudoku(matriz, titulo, matriz_original=None):

    html = f"""
    <html>
    <head>
    <style>

    body {{
        background-color: transparent;
        color: white;
        font-family: Arial, sans-serif;
    }}

    h2 {{
        text-align: center;
    }}

    table {{
        border-collapse: collapse;
        margin: auto;
        border: 4px solid white;
    }}

    td {{
        width: 50px;
        height: 50px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        border: 1px solid #555;
    }}

    .right {{
        border-right: 4px solid white !important;
    }}

    .bottom {{
        border-bottom: 4px solid white !important;
    }}

    </style>
    </head>
    <body>

    <h2>{titulo}</h2>

    <table>
    """

    for i in range(9):

        html += "<tr>"

        for j in range(9):

            clases = ""

            if j in [2, 5]:
                clases += " right"

            if i in [2, 5]:
                clases += " bottom"

            color = "white"

            if matriz_original is not None:
                if matriz_original[i][j] == 0:
                    color = "#4CAF50"

            html += f"""
            <td class="{clases}" style="color:{color};">
                {matriz[i][j]}
            </td>
            """

        html += "</tr>"

    html += """
    </table>
    </body>
    </html>
    """

    components.html(
        html,
        height=560,
        scrolling=False
    )


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
                
                st.write("### 🔍 Lo que la IA está leyendo realmente:")
                st.dataframe(tablero_detectado)


                # =====================
                # MODELO 3 (BACKTRACKING) - Solución
                # =====================
                
                solucion = resolver_sudoku(tablero_detectado)

                
                tablero_vis = tablero_detectado.tolist()
                solucion_vis = solucion.tolist()

            st.success("Sudoku resuelto correctamente")

            col1, col2 = st.columns(2)

            with col1:
                mostrar_sudoku(
                    tablero_vis,
                "Sudoku detectado"
    )

            with col2:
                mostrar_sudoku(
                    solucion_vis,
                "Solución",
                tablero_vis
    )

        except Exception as e:

            if "No se ha encontrado solución" in str(e):

                st.warning(
            """
            🟠 SuGoku no ha podido resolver este tablero.

            Parece que algunos números no se han detectado correctamente.

            Recomendaciones:
            • Utiliza una imagen más nítida.
            • Evita sombras o reflejos.
            • Asegúrate de que el Sudoku aparece completo.
            • Prueba con otro Sudoku.
            """
        )

            else:

                st.error(f"Error durante el proceso: {e}")