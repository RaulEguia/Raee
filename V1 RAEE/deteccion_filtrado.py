import cv2
import math
import time
from ultralytics import YOLO

# Inicializar el modelo YOLO
model = YOLO('best.pt')
clsName = ['Plastic', 'Paper', 'Electronic']  # Asegúrate de que coincida con el modelo

# Configuración de la cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Establecer un umbral de confianza
confidence_threshold = 0.85  # Detecciones con al menos un 85% de certeza

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo acceder a la cámara.")
            break

        # Procesar el frame para el modelo YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb, stream=True, verbose=False)

        # Variable para indicar si se detectó un objeto con alta confianza
        detected = False

        # Procesar los resultados
        for res in results:
            boxes = res.boxes
            for box in boxes:
                cls = int(box.cls[0])  # Clase del objeto detectado
                conf = box.conf[0]  # Precisión de la detección

                # Solo mostrar detecciones que superen el umbral de confianza
                if conf >= confidence_threshold:
                    if cls < len(clsName):
                        detected = True
                        # Mostrar por consola
                        print(f"Se detectó {clsName[cls]} con una precisión de {math.ceil(conf * 100)}%")
                        print("=========================================================================")

        # Si no se detectó ningún objeto con suficiente confianza
        if not detected:
            print("No se detectó ningún objeto.")
        
        # Pausa de 5 segundos entre detecciones
        time.sleep(3)
                
except KeyboardInterrupt:
    print("Detección interrumpida por el usuario.")

finally:
    cap.release()
    print("Cámara liberada.")
