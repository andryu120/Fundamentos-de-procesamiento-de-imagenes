import numpy as np
import cv2
from procesamiento import (image, mostrar_imagen, transformacion,clahe, imhist, mostrar_region_homogenea, transformacion_bin)
import matplotlib.pyplot as plt
import os
import skimage

path = os.getcwd()

gravel = skimage.data.gravel()


#Funciones para el analisis

def probar_distancias(img):
    for i in [64,32,16]:
        print("analisis distancias")
        img_modificada = transformacion(img, distancia_y=i, distancia_x=i, alto_region=128, ancho_region=128, control_limite= None)
        mostrar_imagen(img_modificada)

def probar_overlap(img):
    for i in [64,128,256,512]:
        print("analisis overlap")
        img_modificada = transformacion(img, distancia_y=64, distancia_x=64, alto_region=i, ancho_region=i, control_limite= None)
        mostrar_imagen(img_modificada)

def probar_bin(img):
    for i in [64,32,16]:
        img_modificada = transformacion_bin(img, distancia_y=64, distancia_x=64, alto_region=128, ancho_region=128, control_limite= None, n_bins=i)
        mostrar_imagen(img_modificada)

def probar_limite(img):
    for i in [1,0.05,0.01,0]:
        img_modificada = transformacion(img, distancia_y=64, distancia_x=64, alto_region=128, ancho_region=128, control_limite= i)
        mostrar_imagen(img_modificada)
        imhist(img_modificada)


def comparacion(img,dist, reg, limite, clipLimiter, tileGridSize):
    
    img_modificada = transformacion(img, distancia_y=dist, distancia_x=dist, alto_region=reg, ancho_region=reg, control_limite= limite)
    mostrar_imagen(img_modificada)
    imhist(img_modificada)
    img_clahe = clahe(img, clipLimiter, tileGridSize)
    mostrar_imagen(img_clahe)
    imhist(img_clahe)

def color_lab(name,path):
    for root, dirs, files in os.walk(path):
        if name in files:
            ruta = os.path.join(root, name)
    name = ruta
    img_bgr = cv2.imread(name, cv2.IMREAD_COLOR)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(img_lab)

    L_eq = transformacion(L, distancia_y=64, distancia_x=64, alto_region=128, ancho_region=128, control_limite=0.01)
    L_eq = np.clip(L_eq, 0, 255).astype(np.uint8)

    img_procesada = cv2.merge((L_eq, a, b))
    img_rgb_procesada = cv2.cvtColor(img_procesada, cv2.COLOR_LAB2RGB)
    plt.imshow(img_rgb_procesada)
    plt.show()


if __name__ == "__main__":
    img = image("P2_IMG_2423.png", path)
    alto_img, ancho_img = img.shape[:2]

    # Aca se hacen todas pruebas, si se quiere realizar alguna, se descomenta la linea deseada

    #1 Ecualizacion global
    # img_global = transformacion(img, distancia_y=alto_img, distancia_x=ancho_img, alto_region=alto_img, ancho_region=ancho_img, control_limite=None)
    # mostrar_imagen(img_global)
    # imhist(img_global)

    #2,3 Pruebas de bins, malla, overlap y region homogenea
    # probar_distancias(img)
    # probar_overlap(img)
    # probar_bin(img)
    # mostrar_region_homogenea(img, distancia_y=64, distancia_x=64, alto_region=128, ancho_region=128)

    #4 explorar limites
    # probar_limite(img)
    # probar_limite(gravel)
    
    #5 Comparacion entre CLAHE y metodo propio
    # Config 1 
    # comparacion(gravel, dist=64, reg=128, limite=0.01, clipLimiter=2.0, tileGridSize=(16,16))
    # Config 2 
    # comparacion(gravel, dist=64, reg=128, limite=0.1, clipLimiter=10.0, tileGridSize=(16,16))
    
    # 6. Macro vs Micro (regiones)
    # Micro:
    # comparacion(img, dist=16, reg=32, limite=0.1, clipLimiter=2.0, tileGridSize=(32,32))
    # Macro:
    # comparacion(img, dist=128, reg=256, limite=0.1, clipLimiter=2.0, tileGridSize=(4,4))
    

    
    #7 Discontinuidades
    # NOTA: Para probar esto, se tiene que modificar temporalmente procesamiento.py
    # forzando los pesos a: peso_y1 = 1.0, peso_y2 = 0.0, peso_x1 = 1.0, peso_x2 = 0.0
    # img = transformacion(img, distancia_y=64, distancia_x=64, alto_region=128, ancho_region=128, control_limite=None)
    # mostrar_imagen(img)


    #8 CIELAB 
    # color_lab("imagen.jpg", path)
    
    


    

    

    
    
    

    
    

    
