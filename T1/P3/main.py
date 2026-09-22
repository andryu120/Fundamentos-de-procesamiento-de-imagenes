
from procesamiento import (image,escalamiento,guardar_imagen,bicubica_interpolation)
import cv2
import numpy as np
import skimage

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
    print(f'peso_Arriba-Izq: {peso_x1*peso_y1}')
    print(f'peso_Arriba-Der: {peso_x2*peso_y1}')
    print(f'peso_Abajo-Izq: {peso_x1*peso_y2}')
    print(f'peso_Abajo-Der: {peso_x2*peso_y2}')

    print(f"Valor final interpolado: {pixel_resultante}")

gravel = skimage.data.gravel()
clahe_cv2 = cv2.createCLAHE(4.0, (8,8))
gravel = clahe_cv2.apply(gravel)
guardar_imagen(gravel, "gray", "gravel")


if __name__ == "__main__":
    img = image("P3_IMG_2387_crop.tif","rgb")
    
    #1 a 4
    #
    # for i in [0.5,0.8,1.5,1.9]:
    #     imagen_modificada = escalamiento(img,"bilineal",i)
    #     imagen_modificada_gravel = escalamiento(gravel,"bilineal",i)
    #     guardar_imagen(imagen_modificada,"rgb" ,f' Imagen Tigre, Escalamiento bilineal con parametro s = {i}')
    #     guardar_imagen(imagen_modificada_gravel,"gray" ,f'Imagen Gravel, Escalamiento bilineal con parametro s = {i}')
    #     imagen_modificada = escalamiento(img,"vecino",i)
    #     imagen_modificada_gravel = escalamiento(gravel,"vecino",i)
    #     guardar_imagen(imagen_modificada,"rgb" ,f'Imagen Tigre, Escalamiento vecino mas cercano con parametro s = {i}')
    #     guardar_imagen(imagen_modificada_gravel,"gray" ,f'Imagen Gravel, Escalamiento vecino mas cercano con parametro s = {i}')
    # # 5
    # img_reducida = escalamiento(gravel,"bilineal",0.5)
    # img_ampliada = escalamiento(img_reducida,"bilineal",2)
    # guardar_imagen(img_ampliada, "gray", "Imagen Gravel, bilineal, imagen recuperada")
    # img_reducida = escalamiento(gravel,"vecino",0.5)
    # img_ampliada = escalamiento(img_reducida,"vecino",2)
    # guardar_imagen(img_ampliada, "gray", "Imagen Gravel,vecino cercano, imagen recuperada")
    #6
    img1 = escalamiento(gravel,"bilineal",1.9)
    img2 = escalamiento(img1,"bilineal",0.8)
    img3 = escalamiento(img2,"bilineal",0.5)
    guardar_imagen(img3, "gray", "Imagen Gravel,bilineal, re escalada s = 0.76")
    img_equivalente  = escalamiento(gravel,"bilineal",0.76)
    guardar_imagen(img_equivalente, "gray", "Imagen Gravel,bilineal, escalada s = 0.76")

    # #7

    # img_aliasing = escalamiento(gravel, "bilineal", 0.1)
    # guardar_imagen(img_aliasing, "gray", "Imagen Gravel, aliasing, s = 0.1")
    # img_aliasing = escalamiento(img, "bilineal", 0.1)
    # guardar_imagen(img_aliasing, "rgb", "Imagen Tigre, aliasing, s = 0.1")
    # img_aliasing = escalamiento(gravel, "vecino", 0.1)
    # guardar_imagen(img_aliasing, "gray", "Imagen Gravel,vecino cercano, aliasing, s = 0.1")
    # img_aliasing = escalamiento(img, "vecino", 0.1)
    # guardar_imagen(img_aliasing, "rgb", "Imagen Tigre,vecino cercano, aliasing, s = 0.1")

    # #8

    # img_bicubica = bicubica_interpolation(gravel, 4.2)
    # img_bilineal = escalamiento(gravel, "bilineal", 4.2)
    # img_vecino = escalamiento(gravel, "vecino", 4.2)
    # guardar_imagen(img_bicubica, "gray", "Imagen Gravel, bicubica , s = 4.2")
    # guardar_imagen(img_bilineal, "gray", "Imagen Gravel, bilineal , s = 4.2")
    # guardar_imagen(img_vecino, "gray", "Imagen Gravel, vecino cercano, s = 4.2")






        

    

