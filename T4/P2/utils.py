import os
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage

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