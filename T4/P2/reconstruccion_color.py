import os
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import (rgb2lab, lab2lch, lch2lab, lab2rgb, lab2xyz)
from   scipy.fft  import fft2, ifft2, fftshift, ifftshift, dctn, idctn
from utils import (luminancia, recolorear, reescalar, a_pantalla)
from durand import durand
from homorfico import homomorfico_hdr

M_XYZ2RGB = np.array([[ 3.2406, -1.5372, -0.4986],
                      [-0.9689,  1.8758,  0.0415],
                      [ 0.0557, -0.2040,  1.0570]])

def re_color(E, Y, L_out, factor):

    
    # pasar a CIELAB y trabajar en C
    # idea: cambiar la luminancia con C con cierto porcentaje (parametro), luego normalizar y pasar a RGB
    ref = a_pantalla(E / np.percentile(Y, 99))
    C_og = lab2lch(rgb2lab(ref))
    #C_og = lab2lch(rgb2lab(a_pantalla(E))) # de la imagen original

    L_mod = rgb2lab(np.stack([a_pantalla(L_out)] * 3, axis=-1))[..., 0] # esta es la luminancia modificada # revisar esto

    lch_mod = np.stack([L_mod, C_og[:,:,1] * factor, C_og[:,:,2]], axis=-1)
    lab_mod = lch2lab(lch_mod)

    rgb_lin = lab2xyz(lab_mod) @ M_XYZ2RGB.T
    
    #new_image = lab2rgb(lch2lab((lch_mod))) # aca se reconstruye la imagen con el nuevo espacio modificado
    fuera_de_rango = np.mean(np.any((rgb_lin < -1e-3) | (rgb_lin > 1 + 1e-3), axis=-1))# esto es para los pixeles fuera de rango 

    return lab2rgb(lab_mod) , fuera_de_rango

