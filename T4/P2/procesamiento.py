import os
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from   scipy.fft  import fft2, ifft2, fftshift, ifftshift, dctn, idctn
from homorfico import homomorfico_hdr, filtro_homomorfico
from durand import durand
from utils import (reescalar, luminancia)


def cargar_rgb(path):   
    """
    Carga una imagen HDR y devuelve RGB lineal float32 normalizado por su máximo.
    cv2 entrega BGR, por eso se invierte el orden de canales: luminancia()
    asume que el canal 0 es R.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    rgb = img[..., :3][..., ::-1].astype(np.float32)
    return rgb / rgb.max()


# funciones especificas para por_canal

def homomorfico_2d(X, gamma_H=1.5, gamma_L=0.25, D0=40, c=1.0):
    
    return reescalar(filtro_homomorfico(X, gamma_H, gamma_L, D0, c), 0.5, 99)

def homomorfico_hdr(E, **kw):
    Y = luminancia(E)
    return Y, homomorfico_2d(Y, **kw)


def durand_2d(E, sigma_s=None, sigma_r=0.4, contraste=30, s=0.6, filtro='bilateral'):
    Y = luminancia(E)
    logY = np.log10(Y + 1e-12).astype(np.float32)
    if sigma_s is None:
        sigma_s = 0.02 * max(Y.shape)                         # 2% del tamaño de la imagen

    if filtro == 'bilateral':
        base = cv2.bilateralFilter(logY, d=-1, sigmaColor=sigma_r, sigmaSpace=sigma_s)
    else:
        base = cv2.GaussianBlur(logY, (0, 0), sigma_s)
    detalle = logY - base

    k = np.log10(contraste) / (base.max() - base.min())
    L_out = 10 ** (k * (base - base.max()) + detalle)
    return np.clip(L_out, 0.0, 1.0)

def durand(E, **kw): #?
    Y = luminancia(E)
    return Y, durand_2d(Y, **kw)


def por_canal(img, metodo: str):
    # Verificar si se trabaja en RGB o BGR
    salida = []

    for c in range(3):
        C = np.ascontiguousarray(img[..., c])
        if metodo == "homomorfico":
            salida.append(homomorfico_2d(C))
        elif metodo == "durand":
            salida.append(durand_2d(C))

    return np.clip(np.stack(salida, axis=-1), 0.0, 1.0) # normalizado a [0,1]
