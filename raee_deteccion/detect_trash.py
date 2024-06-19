# Libraries
from tkinter import *
from PIL import Image, ImageTk
import imutils
import cv2
import numpy as np
from ultralytics import YOLO
import math

def clean_lbl():
    # Clean
    lblimg.config(image='')
    lblimgtxt.config(image='')

def images(img, imgtxt):
    img = img
    imgtxt = imgtxt

    # Img Detect
    img = np.array(img, dtype="uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = Image.fromarray(img)

    img_ = ImageTk.PhotoImage(image=img)
    lblimg.configure(image=img_)
    lblimg.image = img_

    # Img Text
    imgtxt = np.array(imgtxt, dtype="uint8")
    imgtxt = cv2.cvtColor(imgtxt, cv2.COLOR_BGR2RGB)
    imgtxt = Image.fromarray(imgtxt)

    img_txt = ImageTk.PhotoImage(image=imgtxt)
    lblimgtxt.configure(image=img_txt)
    lblimgtxt.image = img_txt

# Scanning Function
def Scanning():
    global img_plastic, img_paper, img_electronic
    global img_plastictxt, img_papertxt, img_electronicxt, pantalla
    global lblimg, lblimgtxt

    # Interfaz
    lblimg = Label(pantalla)
    lblimg.place(x=75, y=260)

    lblimgtxt = Label(pantalla)
    lblimgtxt.place(x=995, y=310)
    detect = False

    # Read VideoCapture
    if cap is not None:
        ret, frame = cap.read()
        frame_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # True
        if ret:
            # Yolo | AntiSpoof
            results = model(frame, stream=True, verbose=False)
            for res in results:
                # Box
                boxes = res.boxes
                for box in boxes:
                    detect = True
                    # Bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Error < 0
                    if x1 < 0: x1 = 0
                    if y1 < 0: y1 = 0
                    if x2 < 0: x2 = 0
                    if y2 < 0: y2 = 0

                    # Class
                    cls = int(box.cls[0])

                    # Confidence
                    conf = math.ceil(box.conf[0] * 100)

                    if cls == 0:
                        # Draw
                        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red
                        # Text
                        text = f'{clsName[cls]} {conf}%'
                        sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        dim = sizetext[0]
                        baseline = sizetext[1]
                        # Rect
                        cv2.rectangle(frame_show, (x1, y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline), (0, 0, 0), cv2.FILLED)
                        cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        # Clasificacion
                        images(img_plastic, img_plastictxt)

                    if cls == 1:
                        # Draw
                        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green
                        # Text
                        text = f'{clsName[cls]} {conf}%'
                        sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        dim = sizetext[0]
                        baseline = sizetext[1]
                        # Rect
                        cv2.rectangle(frame_show, (x1, y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline), (0, 0, 0), cv2.FILLED)
                        cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        # Clasificacion
                        images(img_paper, img_papertxt)

                    if cls == 2:
                        # Draw
                        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Cyan
                        # Text
                        text = f'{clsName[cls]} {conf}%'
                        sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        dim = sizetext[0]
                        baseline = sizetext[1]
                        # Rect
                        cv2.rectangle(frame_show, (x1, y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline), (0, 0, 0), cv2.FILLED)
                        cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                        # Clasificacion
                        images(img_electronic, img_electronicxt)

            if detect == False:
                # Clean
                clean_lbl()

            # Resize
            frame_show = imutils.resize(frame_show, width=640)

            # Convertimos el video
            im = Image.fromarray(frame_show)
            img = ImageTk.PhotoImage(image=im)

            # Mostramos en el GUI
            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10, Scanning)

        else:
            cap.release()

# main
def ventana_principal():
    global cap, lblVideo, model, clsName, img_plastic, img_paper, img_electronic
    global img_plastictxt, img_papertxt, img_electronicxt, pantalla

    # Ventana principal
    pantalla = Tk()
    pantalla.title("RECICLAJE INTELIGENTE")
    pantalla.geometry("1280x720")

    # Background
    imagenF = PhotoImage(file="setUp/UCSM.png")
    background = Label(image=imagenF, text="Inicio")
    background.place(x=0, y=0, relwidth=1, relheight=1)

    # Clases: 0 -> Plásticos | 2 -> Papel | 3 -> Electrónico
    # Model
    model = YOLO('best.pt')

    # Clases
    clsName = ['Plastic', '', 'Paper', 'Electronic']

    # Images
    img_plastic = cv2.imread("setUp/plastico.png")
    img_paper = cv2.imread("setUp/papel.png")
    img_electronic = cv2.imread("setUp/electronico.png")

    # Text Images
    img_plastictxt = cv2.imread("setUp/plasticotxt.png")
    img_papertxt = cv2.imread("setUp/papeltxt.png")
    img_electronicxt = cv2.imread("setUp/electronicotxt.png")

    # Video
    lblVideo = Label(pantalla)
    lblVideo.place(x=320, y=180)

    # Elegimos la camara
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 720)
    cap.set(4, 480)
    Scanning()

    # Eject
    pantalla.mainloop()

if __name__ == "__main__":
    ventana_principal()
