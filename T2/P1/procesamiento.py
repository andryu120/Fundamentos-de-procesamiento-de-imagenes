import numpy as np
import cv2

import scipy

#creacion imagen

def mostrar_imagen(img: np.ndarray):
    import matplotlib.pyplot as plt
    # # Si queremos mostrala
    plt.figure(figsize=(15,8))
    plt.imshow(img, cmap='gray')
    plt.show()


def kernel_gaussiano(sigma: int):
    
    # el valor recomendado para el soporte del filtro es 4*sigma + 1 
    size = 2*(int(np.ceil(2*sigma))) + 1
    
    center = size // 2
    x, y = np.mgrid[-center:center+1, -center:center+1]

    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()


    return kernel

def filtro_gaussiano(imagen, sigma: int):
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


def filtro_gaussiano_adap(imagen):
    pass

def suaviado_gauss():
    pass
