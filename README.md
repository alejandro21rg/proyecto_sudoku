# proyecto_sudoku
#  SuGoku Solver

### Resolución Inteligente de Sudokus mediante Visión Artificial y Deep Learning
---

# 📖 Descripción

**SuGoku Solver** es una aplicación basada en Inteligencia Artificial capaz de detectar, interpretar y resolver automáticamente sudokus a partir de una fotografía o imagen proporcionada por el usuario.


#  Objetivos del Proyecto

* Detectar automáticamente tableros de sudoku en imágenes.
* Corregir la perspectiva del tablero.
* Reconocer los números existentes mediante una CNN.
* Generar la matriz inicial del sudoku.
* Resolver el sudoku utilizando Inteligencia Artificial.
* Mostrar la solución mediante una interfaz web intuitiva.

---

#  Arquitectura General

```text
Imagen
   │
   ▼
Modelo 1 (YOLO)
Detección del tablero
   │
   ▼
Modelo 2 (CNN OCR)
Reconocimiento de dígitos
   │
   ▼
Modelo 3 (Deep Learning + Solver)
Predicción y resolución
   │
   ▼
Resultado final
```

---

# Modelo 1 - Detección del Sudoku

## Función

Localizar automáticamente el tablero dentro de una imagen.

## Tecnologías

* YOLO
* OpenCV
* NumPy

## Proceso

1. Recepción de imagen.
2. Detección del tablero.
3. Recorte automático.
4. Corrección de perspectiva.
5. Obtención del tablero cuadrado.

## Resultado

Imagen limpia y preparada para el OCR.

---

#  Modelo 2 - OCR mediante CNN

## Función

Reconocer los dígitos presentes en cada una de las 81 celdas.

## Tecnologías

* TensorFlow
* Keras
* Redes Neuronales Convolucionales (CNN)

## Arquitectura CNN

```text
Input (28x28x1)
      │
Conv2D
      │
MaxPooling
      │
Conv2D
      │
MaxPooling
      │
Flatten
      │
Dense
      │
Dense (10 clases)
```

## Proceso

1. División del tablero en 81 celdas.
2. Conversión a escala de grises.
3. Binarización.
4. Normalización.
5. Clasificación del dígito.
6. Generación de la matriz del sudoku.

## Resultado

```text
[
 [5,3,0,0,7,0,0,0,0],
 [6,0,0,1,9,5,0,0,0],
 ...
]
```

---

#  Modelo 3 - Resolución mediante Deep Learning

## Función

Predecir y resolver la solución del sudoku a partir de la matriz obtenida por el OCR.

## Tecnologías

* TensorFlow
* Redes Neuronales Profundas
* NumPy

## Entradas

Matriz 9x9 del sudoku incompleto.

## Salidas

Matriz 9x9 completamente resuelta.

## Arquitectura

```text
Input (81 valores)
       │
Dense
       │
Dense
       │
Dense
       │
Output (81 valores)
```

## Objetivo

Aprender patrones y restricciones propias de los sudokus mediante ejemplos previamente resueltos.

---

#  Algoritmo de Backtracking

Además del modelo de Deep Learning, se implementa un algoritmo clásico de Backtracking.

## Funciones

* Verificación de filas.
* Verificación de columnas.
* Verificación de subcuadrículas.
* Búsqueda recursiva de soluciones.

## Ventajas

✅ Garantiza una solución válida.

✅ Corrige posibles errores del modelo neuronal.

✅ Permite comparar resultados entre IA y métodos tradicionales.

---

# Estructura del Proyecto

```text
proyecto_sudoku/
│
├── .git/
│
├── app/
│
├── celdas/
│
├── modelo_class/
│   ├── __pycache__/
│   ├── celdas/
│   ├── celdas_prueba/
│   ├── modelo_2.ipynb
│   ├── modelo_2.py
│   ├── modelo_claass.ipynb
│   ├── modelo2_cnn.keras
│   ├── sudoku_detectado.npy
│   └── sudoku_detectado.txt
│
├── modelo_juego/
│   ├── __pycache__/
│   ├── data/
│   ├── backtracking.ipynb
│   ├── deep.py
│   ├── modelo_3.py
│   ├── prueba_1.ipynb
│   └── sudoku_solver.keras
│
├── modelo_yolo/
│   ├── __pycache__/
│   ├── img_pro/
│   ├── runs/
│   ├── best.pt
│   ├── best_1.pt
│   ├── best_2.pt
│   ├── data.yaml
│   ├── imagen_recortada.png
│   ├── imagen_recortada2.png
│   ├── modelo_1.ipynb
│   ├── modelo_1.py
│   ├── prueba_11.png
│   ├── prueba_12.png
│   ├── prueba_13.png
│   ├── prueba_sudoku.png
│   ├── prueba_sudoku_1.jpg
│   ├── sudoku_online.png
│   └── yolov8n.pt
│
├── modelos/
│   ├── cnn_1m.keras
│   └── info_modelo.md
│
├── app.py
├── app2.py
├── app2_deep.py
└── README.md
```

---

# ⚙️ Tecnologías Utilizadas

| Tecnología | Uso                       |
| ---------- | ------------------------- |
| Python     | Desarrollo principal      |
| OpenCV     | Procesamiento de imágenes |
| YOLO       | Detección del tablero     |
| TensorFlow | Deep Learning             |
| Keras      | Redes neuronales          |
| NumPy      | Operaciones matriciales   |
| Streamlit  | Aplicación web            |
| Pillow     | Manipulación de imágenes  |

---

# 🚀 Instalación

## 1. Clonar repositorio

```bash
git clone https://github.com/usuario/sugoku-solver.git
cd sugoku-solver
```

## 2. Crear entorno virtual

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

## 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

# ▶️ Ejecución

```bash
streamlit run app.py
```

La aplicación estará disponible en:

```text
http://localhost:8501
```

---

# 📸 Flujo Completo

```text
Imagen subida
      │
      ▼
YOLO detecta tablero
      │
      ▼
Corrección perspectiva
      │
      ▼
División en 81 celdas
      │
      ▼
CNN reconoce números
      │
      ▼
Generación matriz
      │
      ▼
Deep Learning Solver
      │
      ▼
Backtracking Validación
      │
      ▼
Sudoku resuelto
```

---

# Resultados

El sistema es capaz de:

* Detectar correctamente tableros de sudoku.
* Reconocer dígitos con alta precisión.
* Resolver sudokus automáticamente.
* Procesar imágenes en pocos segundos.
* Mostrar resultados de forma intuitiva.

---

# Limitaciones

* Fotografías con baja iluminación.
* Imágenes desenfocadas.
* Sudokus parcialmente ocultos.
* Escritura manual compleja.

---



