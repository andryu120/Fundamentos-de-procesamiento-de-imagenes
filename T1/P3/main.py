
from procesamiento import (image,escalamiento,guardar_imagen)
import cv2
import numpy as np


def selecionar_pixel(img, s: float, x: int, y: int):
    alto_imagen, ancho_imagen = img.shape[:2]
    alto_new_image = int(alto_imagen*s)
    ancho_new_image = int(ancho_imagen*s)
    # detectar si es en escala de grises o rgb
    if len(img.shape) == 3:
        canales = img.shape[2]
        new_image = np.zeros((alto_new_image, ancho_new_image, canales))
    else:
        new_image = np.zeros((alto_new_image, alto_new_image))
    x_entrada = x/s
    y_entrada = y/s
    x1 = int(x_entrada)
    x2 = x1 + 1
    y1 = int(y_entrada)
    y2 = y1 + 1

    #verificar que estan dentro del rango
    if y2 >= alto_imagen:
        peso_y1 = 1.0
        peso_y2 = 0.0
    else:
        peso_y1 = (y2-y_entrada)/(y2-y1)
        peso_y2 = (y_entrada-y1)/(y2-y1)
    if  x2 >= ancho_imagen:
        peso_y1 = 1.0
        peso_y2 = 0.0
    else:
        peso_x1 = (x2-x_entrada)/(x2-x1)
        peso_x2 = (x_entrada-x1)/(x2-x1)

    p11= img[y1,x1] #arriba_izq 
    p12= img[y1,x2] #arriba_der 
    p21 = img[y2,x1] #abajo_izq
    p22 = img[y2,x2] #arriba_der 

    pixel_resultante = (p11*peso_y1*peso_x1)+(p12*peso_y1*peso_x2) + (p21*peso_y2*peso_x1) + (p22*peso_y2*peso_x2)

    print(f"Coordenada Salida ({x}, {y})")
    print(f'Coordenda entrada ({x_entrada},{y_entrada})')
    print(f"Cuatro vecinos: \n")
    print(f"  Arriba-Izq : ({x1}, {y1}) -> Valor: {p11}")
    print(f"  Arriba-Der : ({x2}, {y1}) -> Valor: {p12}")
    print(f"  Abajo-Izq  : ({x1}, {y2}) -> Valor: {p21}")
    print(f"  Abajo-Der  : ({x2}, {y2}) -> Valor: {p22}")
    print("\n")
    print("Pesos:")
    print(f'peso_x1: {peso_x1}')
    print(f'peso_y1: {peso_y1}')
    print(f'peso_x2: {peso_x2}')
    print(f'peso_y2: {peso_y2}')

    print(f"Valor final interpolado: {pixel_resultante}")



if __name__ == "__main__":
    img = image("P3_IMG_2387_crop.jpg","rgb")
   
    img_modificada = escalamiento(img, "bilineal", 0.6)

    guardar_imagen(img_modificada, "rgb", 'prueba')