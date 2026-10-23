import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
import os
from   scipy.fft     import fft, fft2, ifft2, fftshift, ifftshift, fftfreq
from   scipy.ndimage import gaussian_filter, convolve, convolve1d
from   scipy.signal  import fftconvolve
from   scipy.signal.windows import tukey



#Importar una imagen
def image(name):
    path = os.getcwd()
    for root, dirs, files in os.walk(path):
        if name in files:
            ruta = os.path.join(root, name)
    name = ruta
    imagen = cv2.imread(name, cv2.IMREAD_GRAYSCALE) # leer en escalas de grises
    return imagen


def fourier(imagen):
    rows, cols = imagen.shape
    return fftshift(fft2(imagen))


# filtro ideal


# filtro gaussiano