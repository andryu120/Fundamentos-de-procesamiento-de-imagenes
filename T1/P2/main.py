import numpy as np
import cv2
from procesamiento import (image, mostrar_imagen, transformacion,clahe, imhist, mostrar_region_homogenea, transformacion_bin)

import os
path = os.getcwd()


if __name__ == "__main__":
    img = image("P2_IMG_2423.png", path)
    alto_img, ancho_img = img.shape[:2]
    #imhist(img)
    img_modificada = transformacion(img, distancia_y=alto_img, distancia_x=ancho_img, alto_region=alto_img, ancho_region=ancho_img, control_limite= 0.0)
    #mostrar_imagen(img_modificada)
    imhist(img_modificada)

    #img_modificada = transformacion(img, distancia_y=64, distancia_x=64, alto_region=128, ancho_region=128, control_limite= 0.0)
    #mostrar_imagen(img_modificada)

    #mostrar_region_homogenea(img,distancia_y=64, distancia_x=64, alto_region=128, ancho_region=128)

    #img_clahe = clahe(img)
    #mostrar_imagen(img_clahe)

    # for i in [64,32,16]:
    #     print("analisis distancias")
    #     img_modificada = transformacion(img, distancia_y=i, distancia_x=i, alto_region=128, ancho_region=128, control_limite= 0.0)
    #     mostrar_imagen(img_modificada)

    # for i in [64,128,256,512]:
    #     print("analisis overlap")
    #     img_modificada = transformacion(img, distancia_y=64, distancia_x=64, alto_region=i, ancho_region=i, control_limite= 0.0)
    #     mostrar_imagen(img_modificada)

    # for i in [64,32,16]:
    #     img_modificada = transformacion_bin(img, distancia_y=64, distancia_x=64, alto_region=128, ancho_region=128, control_limite= 0.0, n_bins=i)
    #     mostrar_imagen(img_modificada)

    
    

    
    

    
