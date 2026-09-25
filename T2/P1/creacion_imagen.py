import numpy as np
import cv2





alto, ancho = 256,256
size_rectangle = 128
radius_circle = 32

image = np.zeros((256,256,1),dtype=np.uint8)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # conversion de BGR a RGB

y,x = np.ogrid[:alto,:ancho]
#hacer las mascaras
#limites
limite_inferior = size_rectangle - (size_rectangle / 2) 
limite_superior = size_rectangle + (size_rectangle / 2) 
mascara_rectangulo = (x >= limite_inferior) & (x <= limite_superior) & (y >= limite_inferior) & (y <= limite_superior)
mascara_exterior = (x < limite_inferior) | (x > limite_superior) | (y < limite_inferior) | (y > limite_superior)
#mascara circulo
mascara_circulo = ((x - ancho/2)**2 +(y - alto/2)**2 <= radius_circle**2)

image[mascara_rectangulo] = 255.0*0.45
image[mascara_circulo] = 255.0*0.8
image[mascara_exterior] = 255.0*0.15
                

        



def agregar_ruido_poisson(imagen, lam):
    # El ruido de Poisson depende de la señal: donde hay más intensidad,
    # más varianza. Escalamos la imagen a un "conteo de fotones" (lambda, 'lam'),
    # muestreamos, y volvemos al rango [0, 255]
    # OJO: acá lam más BAJO significa MÁS ruido

    imagen = imagen/255.0
    imagen_ruidosa = np.random.poisson(imagen * lam) / lam
    imagen_ruidosa = imagen_ruidosa*255
    imagen_ruidosa = np.clip(imagen_ruidosa, 0, 255)
    imagen_ruidosa = imagen_ruidosa.astype(np.uint8)
    return imagen_ruidosa

imagen_ruidosa = agregar_ruido_poisson(image,40)
