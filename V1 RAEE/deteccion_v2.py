import cv2
import math
from ultralytics import YOLO

# Inicializar el modelo YOLO
model = YOLO('best.pt')
clsName = ['Plastic', 'Paper', 'Electronic']

# Configuración de la cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Establecer un umbral de confianza
confidence_threshold = 0.3  # Ajusta este valor si es necesario

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Procesar el frame para el modelo YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb, stream=True, verbose=False)
        
        # Procesar los resultados
        for res in results:
            boxes = res.boxes
            for box in boxes:
                cls = int(box.cls[0])  # Clase del objeto detectado
                conf = box.conf[0]  # Precisión de la detección
                
                # Solo mostrar detecciones que superen el umbral de confianza
                if conf >= confidence_threshold:
                    if cls < len(clsName):
                        # Mostrar por consola
                        print(f"Se detectó {clsName[cls]} con una precisión de {math.ceil(conf * 100)}%")
                else:
                    # Imprimir información de detección con baja confianza para depuración
                    print(f"Clase {cls} detectada con baja confianza: {math.ceil(conf * 100)}%")
                
except KeyboardInterrupt:
    print("Detección interrumpida por el usuario.")

finally:
    cap.release()
    print("Cámara liberada.")
