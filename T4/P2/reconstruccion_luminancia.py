import os
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from   scipy.fft  import fft2, ifft2, fftshift, ifftshift, dctn, idctn
from utils import (luminancia, recolorear, reescalar, a_pantalla)
from durand import durand
from homorfico import homomorfico_hdr


def re_lum(E, Y, L_out, beta, epsilon = 1e-4): # luminancia procesada, imagen
        

    salida = []

    for c in range(3):
        C = E[..., c]
        salida.append(L_out * ((C/(Y+epsilon)) ** beta))
        

    return np.stack(salida, axis=-1)





