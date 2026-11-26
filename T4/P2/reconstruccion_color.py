import os
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import (rgb2lab, lab2lch, lch2lab, lab2rgb)
from   scipy.fft  import fft2, ifft2, fftshift, ifftshift, dctn, idctn
from utils import (luminancia, recolorear, reescalar, a_pantalla)
from durand import durand
from homorfico import homomorfico_hdr

def re_color(E, metodo, factor):

    if metodo == "homomorfico":
        Y, L_out = homomorfico_hdr(E)

    elif metodo == "durand":
        Y, L_out = durand(E)
    # pasar a CIELAB y trabajar en C
    # idea: cambiar la luminancia con C con cierto porcentaje (parametro), luego normalizar y pasar a RGB
    C_og = lab2lch(rgb2lab(a_pantalla(E))) # de la imagen og

    L_mod = rgb2lab(np.stack([L_out]*3, axis=-1)) # esta es la luminancia modificada # revisar esto

    lch_mod = np.stack([L_mod[:, :, 0], C_og[:,:,1] * factor, C_og[:,:,2]], axis=-1)
    
    new_image = lab2rgb(lch2lab((lch_mod))) # aca se reconstruye la imagen con el nuevo espacio modificado
    fuera_de_rango = np.mean((new_image < 0.0) | (new_image > 1.0)) # esto es para los pixeles fuera de rango 

    return np.clip(new_image, 0, 1) , fuera_de_rango

