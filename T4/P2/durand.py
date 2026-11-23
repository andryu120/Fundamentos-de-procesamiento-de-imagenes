import os
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from   scipy.fft  import fft2, ifft2, fftshift, ifftshift, dctn, idctn
from procesamiento import (luminancia, recolorear, reescalar, a_pantalla)


def durand(E, sigma_s=None, sigma_r=0.4, contraste=30, s=0.6, filtro='bilateral'):
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
    return Y, L_out
