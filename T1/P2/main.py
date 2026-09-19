import numpy as np
import cv2
from procesamiento import (image, mostrar_imagen, transformacion)

import os
path = os.getcwd()


if __name__ == "__main__":
    img = image("P2_IMG_2423.png", path)
    alto_img, ancho_img = img.shape[:2]
    
    img_modificada = transformacion(img, distancia_y=alto_img, distancia_x=ancho_img, alto_region=alto_img, ancho_region=ancho_img)
    #img_modificada = np.clip(img_modificada, 0, 255).astype(np.uint8)
    mostrar_imagen(img=img_modificada)
    
