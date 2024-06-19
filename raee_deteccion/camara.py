import cv2


# Función para listar los dispositivos de captura disponibles
def listar_camaras(max_index=10):
    index = 0
    arr = []
    while index < max_index:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr


# Listar cámaras disponibles
camaras_disponibles = listar_camaras()
print("Cámaras disponibles:", camaras_disponibles)

# Probar cada cámara disponible
for i in camaras_disponibles:
    print(f"Probando cámara {i}")
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f"No se puede abrir la cámara {i}")
        continue

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"No se puede recibir el frame de la cámara {i}")
            break

        cv2.imshow(f'Video en Vivo - Cámara {i}', frame)

        if cv2.waitKey(1) == 27:  # Presiona ESC para salir
            break

    cap.release()
    cv2.destroyAllWindows()
