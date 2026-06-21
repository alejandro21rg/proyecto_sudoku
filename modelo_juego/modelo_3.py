# MODELO 3 - RESOLUCIÓN DEL SUDOKU

import numpy as np


def es_valido(tablero, fila, col, num):

    for x in range(9):
        if tablero[fila][x] == num:
            return False

    for x in range(9):
        if tablero[x][col] == num:
            return False

    inicio_fila = fila - fila % 3
    inicio_col = col - col % 3

    for i in range(3):
        for j in range(3):
            if tablero[inicio_fila + i][inicio_col + j] == num:
                return False

    return True


def resolver(tablero):

    for fila in range(9):

        for col in range(9):

            if tablero[fila][col] == 0:

                for num in range(1, 10):

                    if es_valido(tablero, fila, col, num):

                        tablero[fila][col] = num

                        if resolver(tablero):
                            return True

                        tablero[fila][col] = 0

                return False

    return True


def resolver_sudoku(tablero):
    """
    Recibe una matriz 9x9 de numpy y devuelve
    el sudoku resuelto.
    """

    tablero = tablero.tolist()

    if resolver(tablero):
        return np.array(tablero)

    raise Exception(
        "No se ha encontrado solución para el sudoku."
    )