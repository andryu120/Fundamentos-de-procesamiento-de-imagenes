import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def image(name, type: str):
    path = os.getcwd()
    for root, dirs, files in os.walk(path):
        if name in files:
            ruta = os.path.join(root, name)
    name = ruta
    if type == "gray":
        imagen = cv2.imread(name, cv2.IMREAD_GRAYSCALE) # leer en escalas de grises
    else:
        Xbgr = cv2.imread(name,cv2.IMREAD_UNCHANGED)
        imagen = cv2.cvtColor(Xbgr, cv2.COLOR_BGR2RGB) # conversion de BGR a RGB

    return imagen


def bilineal_interpolation(img,s: float, A: np.array):
    new_image = np.zeros_like(img)
    alto_new_image, ancho_new_image = s*(img.shape[:2])
    alto_imagen, ancho_imagen = img.shape[:2]
    for y in range(len(new_image)):
        for x in range(len(new_image[y])):
            x_entrada = x/s
            y_entrada = y/s
            x1 = int(x_entrada)
            x2 = x1 + 1
            y1 = int(y_entrada)
            y2 = y1 + 1

            #verificar que estan dentro del rango
            

            #pixeles

            

            
            
    pass

def vecino_cercano_interpolation():
    pass

def escalamiento(img, intepolacion: str):
    if intepolacion == "bilineal":
        pass
        
    elif intepolacion == "vecino":
        pass
    pass