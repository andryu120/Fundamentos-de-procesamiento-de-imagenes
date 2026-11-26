import os
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from   scipy.fft  import fft2, ifft2, fftshift, ifftshift, dctn, idctn
from utils import (luminancia, recolorear, reescalar, a_pantalla)



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
    return Y, L_out