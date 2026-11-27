import os
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from   scipy.fft  import fft2, ifft2, fftshift, ifftshift, dctn, idctn
from homorfico import homomorfico_hdr, filtro_homomorfico
from durand import durand
from utils import (reescalar)


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

def luminancia(E):
    # Fila Y de la matriz RGB -> XYZ usada en la cápsula de espacios de color
    return 0.177 * E[..., 0] + 0.813 * E[..., 1] + 0.011 * E[..., 2]


def H_homomorfico(forma, gamma_H=1.5, gamma_L=0.5, D0=30, c=1.0):
    P, Q = forma
    v, u = np.mgrid[0:P, 0:Q]
    D2 = (u - Q / 2) ** 2 + (v - P / 2) ** 2          # D(u, v)^2, distancia al centro
    return (gamma_H - gamma_L) * (1 - np.exp(-c * D2 / D0 ** 2)) + gamma_L



def filtro_homomorfico(f, gamma_H=1.5, gamma_L=0.5, D0=30, c=1.0, eps=1e-6):
    N, M = f.shape
    z = np.log(f + eps)                                       # ln
    z_pad = np.pad(z, ((0, N), (0, M)), mode='symmetric')     # relleno por reflexión, P = 2N, Q = 2M
    H = H_homomorfico(z_pad.shape, gamma_H, gamma_L, D0, c)
    S = H * fftshift(fft2(z_pad))                             # DFT centrada y filtro
    s = ifft2(ifftshift(S)).real[:N, :M]                      # DFT inversa y recorte
    return np.exp(s)                                          # exp


def homomorfico_hdr(E, gamma_H=1.5, gamma_L=0.25, D0=40, c=1.0, s=0.6):
    Y = luminancia(E)
    L_out = reescalar(filtro_homomorfico(Y, gamma_H, gamma_L, D0, c), 0.5, 99)
    return L_out



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




def por_canal(img, metodo: str):
    
    salida = []
    rgb = ['r','g','b']

    for c, channel in enumerate(rgb):
        C = np.ascontiguousarray(img[..., c])
        if metodo == "homorfico":

            salida.append(filtro_homomorfico(C))
        elif metodo == "durand":
            salida.append(durand_2d(C))
        

    return np.clip(np.stack(salida, axis=-1), 0.0, 1.0) # normalizado a [0,1]
