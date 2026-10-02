import numpy as np
import cv2





alto, ancho = 256,256
size_rectangle = 128
radius_circle = 32

image = np.zeros((256,256),dtype=np.uint8)
imagen_negro = np.zeros((256,256),dtype=np.uint8)


y,x = np.ogrid[:alto,:ancho]
#hacer las mascaras
#limites
limite_inferior = size_rectangle - (size_rectangle / 2) 
limite_superior = size_rectangle + (size_rectangle / 2) 
mascara_rectangulo = (x >= limite_inferior) & (x <= limite_superior) & (y >= limite_inferior) & (y <= limite_superior)

mascara_exterior = (x < limite_inferior) | (x > limite_superior) | (y < limite_inferior) | (y > limite_superior)
#mascara circulo
mascara_circulo = ((x - ancho/2)**2 +(y - alto/2)**2 <= radius_circle**2)

mascara_rectangulo = mascara_rectangulo & ~mascara_circulo

image[mascara_rectangulo] = 255.0*0.45
image[mascara_circulo] = 255.0*0.8
image[mascara_exterior] = 255.0*0.15
                



def agregar_ruido_gaussiano(imagen, sigma):
    imagen = imagen.copy()
    imagen = imagen/255
    ruido = np.random.normal(0, sigma, imagen.shape)
    imagen_ruidosa = imagen + ruido
    imagen_ruidosa = np.clip(imagen_ruidosa, 0, 1) # OJO: Podemos elegir limitar
    # el rango de la imagen, ya que al sumar ruido puede sobrepasar el límite
    # (no pasa en general en la práctica, pero sí es necesario si hacemos datos sintéticos)
    return imagen_ruidosa


np.random.seed(42) #    semilla aleatoria

imagen_ruidosa = agregar_ruido_gaussiano(image,0.05)

# imagen_ruidosa = cv2.cvtColor(imagen_ruidosa, cv2.COLOR_GRAY2RGB)

# mostrar_imagen(imagen_ruidosa)
