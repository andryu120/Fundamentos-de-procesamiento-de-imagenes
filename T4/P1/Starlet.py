import io
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from scipy import fft, ndimage as ndi
from scipy.linalg import hadamard
from PIL import Image, ImageFilter
import cv2

B3 = np.array([1, 4, 6, 4, 1]) / 16
def starlet(imagen, niveles, umbral=False, a=1, k=1):
    capas = []
    c = imagen
    lista_umbral = []
    coef = []
    for j in range(niveles):
        if umbral:
            
            h = np.zeros(4 * 2**j + 1)
            h[::2**j] = B3                       # filtro con 2^j - 1 ceros entre coeficientes
            c_sig = ndi.convolve1d(ndi.convolve1d(c, h, axis=0, mode="mirror"), h, axis=1, mode="mirror")
            d = c - c_sig
            t = a * np.max(np.abs(d))
            mascara = np.abs(d) >= t
            d_hat = np.where(mascara, d, 0.0)
            lista_umbral.append(d_hat)
            coef.append(np.mean(mascara) * 100.0) # en porcentajes
            capas.append(d)
            c = c_sig
        else:

            h = np.zeros(4 * 2**j + 1)
            h[::2**j] = B3                       # filtro con 2^j - 1 ceros entre coeficientes
            c_sig = ndi.convolve1d(ndi.convolve1d(c, h, axis=0, mode="mirror"), h, axis=1, mode="mirror")
            capas.append(k * (c - c_sig))
            c = c_sig
    return capas, c, lista_umbral, coef


def reconstruccion_starlet(capas, c):
    img_final = c

    for niveles in capas:
        img_final = img_final + niveles

    return img_final