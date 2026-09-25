import numpy as np
import cv2

import scipy

#creacion imagen

def mostrar_imagen(img: np.ndarray):
    import matplotlib.pyplot as plt
    # # Si queremos mostrala
    plt.figure(figsize=(15,8))
    plt.imshow(img)
    plt.show()


def kernel_gaussiano(sigma: float):
    size = 4*sigma + 1
    # el valor recomendado para el soporte del filtro es 4*sigma + 1 
    if size % 2 == 0:
        raise ValueError("El tamaño del kernel debe ser impar.")

    center = size // 2
    x, y = np.mgrid[-center:center+1, -center:center+1]

    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()


    return kernel

def filtro_gaussiano(imagen, sigma: int):
    kernel_gauss = kernel_gaussiano(sigma)
    imagen_conv = scipy.signal.convolve2d(imagen, kernel_gauss, mode='same')
    return imagen_conv


def filtro_gaussiano_adap(imagen):
    pass

def suaviado_gauss():
    pass
