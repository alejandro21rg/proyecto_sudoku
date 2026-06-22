# INFORMACIOÓN MODELO PRED DE JUEGO.
## base:
    Es una red neuronal convolucional que recibe una lista de 81 numeros, en la cual los valores vacios se representan como un 0.

    Entrenado con un millon de partidas.

## Entrada de ejemplo:
    lista = [
    6, 4, 0, 0, 3, 0, 0, 0, 7,
    5, 0, 1, 0, 7, 0, 9, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 0,
    0, 0, 4, 9, 0, 8, 0, 6, 0,
    0, 8, 0, 0, 0, 3, 0, 2, 0,
    0, 0, 0, 4, 0, 0, 0, 0, 0,
    4, 0, 0, 1, 5, 7, 0, 3, 0,
    2, 0, 8, 3, 0, 0, 0, 4, 0,
    7, 5, 0, 0, 0, 0, 0, 9, 0
]

## Transformaciones necesarias:

    1. Convertir lista en un array de numpy.
    2. Añadirle una dimension simple al array
        Resultado del shape: (1, 81)

