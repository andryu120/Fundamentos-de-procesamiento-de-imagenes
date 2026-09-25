import numpy as np
import cv2

largo, ancho = 512,512
size_rectangle = 128
radius_circle = 32

image = np.zeros((256,256,1),dtype=np.uint8)

contador = 0
for y in range(largo):
    for x in range(ancho):
        #hacer las mascaras
        mascara_rectangulo = ((size_rectangle - size_rectangle/2)<= x <= (size_rectangle + size_rectangle/2)) & ((size_rectangle - size_rectangle/2)<= y <= (size_rectangle + size_rectangle/2))
        if mascara_rectangulo:
            image[x,y] = 255
        else:
            try:

                image[x,y] = 0
            except IndexError:
                continue

        




cv2.imshow('Display Window', image)

cv2.waitKey(0)

cv2.destroyAllWindows()

def agregar_ruido_poisson(imagen, lam):
    # El ruido de Poisson depende de la señal: donde hay más intensidad,
    # más varianza. Escalamos la imagen a un "conteo de fotones" (lambda, 'lam'),
    # muestreamos, y volvemos al rango [0, 1]
    # OJO: acá lam más BAJO significa MÁS ruido
    imagen_ruidosa = np.random.poisson(imagen * lam) / lam
    imagen_ruidosa = np.clip(imagen_ruidosa, 0, 1)
    return imagen_ruidosa