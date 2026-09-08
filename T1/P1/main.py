
from procesamiento import (mostrar_imagen, color_saturation, image)
import os
import cv2

path = os.getcwd()

p = [(0, 0.0), (120, 2.0), (240, 2.0), (350, 0.0)]
if __name__ == "__main__":
    modo = str(input())
    imagen_original, copia = image("imagen.png",path)
    img = color_saturation(copia, p, modo)
    
    d = mostrar_imagen(img)
    
    mostrar_imagen(imagen_original)