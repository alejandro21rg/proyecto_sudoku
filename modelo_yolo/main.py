
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# APLICAR FUNCION EXPLICADA EN CLASE

model = YOLO("best_1.pt")

results = model("prueba_sudoku_1.jpg")

resultado = results[0]
imagen_anotada = resultado.plot()    

plt.figure(figsize=(8, 10))
plt.imshow(imagen_anotada[:, :, ::-1])  
plt.axis('off')
plt.title('Objetos detectados por YOLO')
plt.show()