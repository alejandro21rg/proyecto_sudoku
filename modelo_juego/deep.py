import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model


RUTA_MODELO = Path(__file__).resolve().parent.parent / "modelos" / "cnn_1m.keras"

modelo = load_model(str(RUTA_MODELO))

print(RUTA_MODELO)
print(RUTA_MODELO.exists())

def resolver_sudoku(tablero):

    entrada = np.array(tablero)

    entrada = entrada.flatten()
    entrada = entrada.reshape(1, 81)

    pred = modelo.predict(
        entrada,
        verbose=0
    )

    solucion = np.argmax(
        pred,
        axis=2
    )

    solucion = solucion.reshape(
        9,
        9
    )

    return solucion

pred = modelo.predict(
    entrada,
    verbose=0
)

print(pred.shape)
print(pred[0][0])