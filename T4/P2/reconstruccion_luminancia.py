import os
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from   scipy.fft  import fft2, ifft2, fftshift, ifftshift, dctn, idctn
from procesamiento import (luminancia, recolorear, reescalar, a_pantalla)
from durand import durand
from homorfico import homomorfico_hdr


def re_lum(E, metodo, beta, epsilon = 0.01): # luminancia procesada, imagen

    if metodo == "homomorfico":
        Y, L_out = homomorfico_hdr(E)
    elif metodo == "durand":
        Y, L_out = durand(E)
        

    salida = []

    for c in range(3):
        C = E[..., c]
        salida.append(L_out * ((C/(Y+epsilon)) ** beta))
        

    return np.stack(salida, axis=-1)





