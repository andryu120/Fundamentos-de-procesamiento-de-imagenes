import numpy as np
import cv2
import scipy
from creacion_imagen import (mascara_circulo, mascara_exterior, mascara_rectangulo)

#creacion imagen

def mostrar_imagen(img: np.ndarray, titulo: str):
    import matplotlib.pyplot as plt
    # # Si queremos mostrala
    
    plt.figure(figsize=(15,8))
    plt.imshow(img, cmap='gray')
    plt.title(f'{titulo}')
    plt.show()


def kernel_gaussiano(sigma: float):
    
    # el valor recomendado para el soporte del filtro es 4*sigma + 1 
    size = 2*(int(np.ceil(2*sigma))) + 1
    
    center = size // 2
    x, y = np.mgrid[-center:center+1, -center:center+1]

    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()


    return kernel

def filtro_gaussiano(imagen, sigma: float):
    imagen = imagen.copy()
    kernel_gauss = kernel_gaussiano(sigma)
    imagen_conv = scipy.signal.convolve2d(imagen, kernel_gauss, mode='same')
    
    return imagen_conv



def rmse(imagen_referencia: np.ndarray, imagen_estimada: np.ndarray) -> float:
    """
    Calcula el RMSE  entre una imagen de referencia y una imagen estimada.

    Parámetros:
        imagen_referencia : np.ndarray, imagen limpia (ground truth)
        imagen_estimada    : np.ndarray, imagen filtrada a evaluar
    """
    referencia = imagen_referencia.astype(np.float64)
    estimada = imagen_estimada.astype(np.float64)
    return float(np.sqrt(np.mean((referencia - estimada) ** 2)))
    


def filtro_gaussiano_adap(imagen, func_sigma,  valores_minimos: dict, interpolado: str):
    imagen = imagen.copy()
    #funcion por partes
    imagen_filtrada = np.zeros_like(imagen)
    intensidades = [255 * 0.15, 255 * 0.45, 255 * 0.80] 

    sigmas_optimos = [valores_minimos["Minimo Exterior"], valores_minimos["Minimo Rectangulo"], valores_minimos["Minimo Circulo"]]
    
    mu = filtro_gaussiano(imagen, func_sigma)
    
    

    imagen_ex = filtro_gaussiano(imagen, valores_minimos["Minimo Exterior"])
    imagen_rect = filtro_gaussiano(imagen, valores_minimos["Minimo Rectangulo"])
    imagen_circ = filtro_gaussiano(imagen, valores_minimos["Minimo Circulo"])


    if interpolado not in ["Si", "si"]:

        mascara_fondo_est = mu < 0.3*255
        mascara_cuadrado_est = (mu >= 0.3*255) & (mu < 0.6*255)
        mascara_circulo_est = mu >= 0.6*255

        imagen_filtrada[mascara_fondo_est] = imagen_ex[mascara_fondo_est]
        imagen_filtrada[mascara_cuadrado_est] = imagen_rect[mascara_cuadrado_est]
        imagen_filtrada[mascara_circulo_est] = imagen_circ[mascara_circulo_est]

        mapeo_sigma_discreto = np.zeros_like(mu)

        
        mapeo_sigma_discreto[mu < 0.3*255] = valores_minimos["Minimo Exterior"]
        mapeo_sigma_discreto[(mu >= 0.3*255) & (mu < 0.6*255)] = valores_minimos["Minimo Rectangulo"]
        mapeo_sigma_discreto[mu >= 0.6*255] = valores_minimos["Minimo Circulo"]

        return imagen_filtrada, mapeo_sigma_discreto 

    else:
        # para interpolar, calculamos los pesos respectivos usando las intensidades
        # primero del fondo al cuadrado
        

        peso_cuadrado_v1 = np.clip((mu - intensidades[0]) / (intensidades[1] - intensidades[0]), 0, 1)
        peso_fondo = 1.0 - peso_cuadrado_v1

        # ahora del cuadrado al circulo

        peso_circulo = np.clip((mu - intensidades[1]) / (intensidades[2] - intensidades[1]), 0, 1)
        peso_cuadrado_v2 = 1.0 - peso_circulo

        # el peso del cuadrado como tal es la interseccion entre ambas regiones
        peso_cuadrado = np.minimum(peso_cuadrado_v1, peso_cuadrado_v2)

        imagen_filtrada = (imagen_ex * peso_fondo) + (imagen_rect * peso_cuadrado) + (imagen_circ * peso_circulo)

        mapeo_sigma_continuo = np.interp(mu.flatten(), intensidades, sigmas_optimos).reshape(mu.shape)

        return imagen_filtrada, mapeo_sigma_continuo
