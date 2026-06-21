from ultralytics import YOLO


modelo = YOLO('yolov8n.pt')


modelo.train (
    # Archivo yaml para definir el dataset.
    data="data.yaml",
    epochs=10,  # Numero de epocas que defino.
    batch=64,   #Tamaño de batch.
    device = "cpu",
    imgsz=640,  # Tamaño de las imagenes.
    project="sudoku",  # Carpeta donde se guardará el entrenamiento.
    name="Sudoku",  # Nombre del modelo.
    save=True,      # Guardar el modelo después de entrenar.
    exist_ok=True   # Sobrescribir resultados si ya existe una carpeta.
    )