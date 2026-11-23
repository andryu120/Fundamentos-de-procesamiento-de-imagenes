import os
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from   scipy.fft  import fft2, ifft2, fftshift, ifftshift, dctn, idctn
from homorfico import homomorfico_hdr
from durand import durand


def cargar_rgb(path):
    """
    Carga una imagen HDR y devuelve RGB lineal float32 normalizado por su máximo.
    cv2 entrega BGR, por eso se invierte el orden de canales: luminancia()
    asume que el canal 0 es R.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    rgb = img[..., :3][..., ::-1].astype(np.float32)
    return rgb / rgb.max()


def luminancia(E):
    # Fila Y de la matriz RGB -> XYZ usada en la cápsula de espacios de color
    return 0.177 * E[..., 0] + 0.813 * E[..., 1] + 0.011 * E[..., 2]


def a_pantalla(x, gamma=2.2):
    # Corrección gamma para desplegar una imagen lineal en [0, 1]
    return np.clip(x, 0.0, 1.0) ** (1.0 / gamma)


def recolorear(E, L_in, L_out, s=0.6):
    # C_out = (C_in / L_in)^s · L_out, para cada canal C = R, G, B
    return np.clip((E / (L_in[..., None] + 1e-12)) ** s * L_out[..., None], 0.0, 1.0)


def reescalar(g, p_bajo=0.5, p_alto=99.5):
    # Lleva los percentiles p_bajo y p_alto a 0 y 1, recortando los extremos
    lo, hi = np.percentile(g, [p_bajo, p_alto])
    return np.clip((g - lo) / (hi - lo), 0.0, 1.0)


def por_canal(img, metodo: str):
    # Verificar si se trabaja en RGB o BGR
    salida = []

    for c in range(3):
        C = np.ascontiguousarray(E[..., c])
        if metodo == "homomorfico":
            salida.append(homomorfico_hdr(C))
        elif metodo == "durand":
            salida.append(durand(C))

    return np.clip(np.stack(salida, axis=-1), 0.0, 1.0) # normalizado a [0,1]
