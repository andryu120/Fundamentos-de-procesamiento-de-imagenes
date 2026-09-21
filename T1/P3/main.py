
from procesamiento import (image,escalamiento,guardar_imagen)
import cv2
import numpy as np


def selecionar_pixel(img):
    
    pass

if __name__ == "__main__":
    img = image("P3_IMG_2387_crop.jpg","rgb")
   
    img_modificada = escalamiento(img, "bilineal", 0.6)

    guardar_imagen(img_modificada, "rgb", 'prueba')