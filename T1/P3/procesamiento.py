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
        return imagen
    else:
        Xbgr = cv2.imread(name,cv2.IMREAD_UNCHANGED)
        imagen = cv2.cvtColor(Xbgr, cv2.COLOR_BGR2RGB) # conversion de BGR a RGB
        return imagen.astype(np.float64)

   


def bilineal_interpolation(img,s: float):
    
    alto_imagen, ancho_imagen = img.shape[:2]
    alto_new_image = int(alto_imagen*s)
    ancho_new_image = int(ancho_imagen*s)
    # detectar si es en escala de grises o rgb
    if len(img.shape) == 3:
       canales = img.shape[2]
       new_image = np.zeros((alto_new_image, ancho_new_image, canales))
    else:
       new_image = np.zeros((alto_new_image, alto_new_image))
    
    for y in range(len(new_image)):
        for x in range(len(new_image[y])):
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
            new_image[y, x] = pixel_resultante  
            
    return new_image

def vecino_cercano_interpolation(img, s: float):
    alto_imagen, ancho_imagen = img.shape[:2]
    alto_new_image = int(alto_imagen*s)
    ancho_new_image = int(ancho_imagen*s)
    # detectar si es en escala de grises o rgb
    if len(img.shape) == 3:
        canales = img.shape[2]
        new_image = np.zeros((alto_new_image, ancho_new_image, canales))
    else:
        new_image = np.zeros((alto_new_image, alto_new_image))
    for y in range(len(new_image)):
        for x in range(len(new_image[y])):
            x_entrada = x/s
            y_entrada = y/s
            x1 = int(x_entrada)
            y1 = int(y_entrada)

            # manejo de bordes
            if x1 >= ancho_imagen:
                x1 = ancho_imagen - 1
            if y1 >= alto_imagen:
                y1 = alto_imagen - 1

            new_image[y,x] = img[y1,x1]

    return new_image
    

def escalamiento(img, intepolacion: str, parametro: float):
    if intepolacion == "bilineal":
        imagen = bilineal_interpolation(img, parametro)
        
        
    elif intepolacion == "vecino":
        imagen = vecino_cercano_interpolation(img, parametro)

    return imagen

def guardar_imagen(img, type: str, titulo: str):
    ruta_carpeta = os.path.join(os.getcwd(), "imagenes")
    nombre_archivo = f"{titulo}.jpg"
    ruta_completa = os.path.join(ruta_carpeta, nombre_archivo)
    if type == "gray":
        cv2.imwrite(ruta_completa, img)
    else:
        # Revertir de RGB a BGR para que cv2.imwrite guarde los colores correctos
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(ruta_completa, img_bgr)


