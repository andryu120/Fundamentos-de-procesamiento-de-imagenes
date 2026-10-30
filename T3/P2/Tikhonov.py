import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
import os
from   scipy.fft     import fft, fft2, ifft2, fftshift, ifftshift, fftfreq
from   scipy.ndimage import gaussian_filter, convolve, convolve1d
from   scipy.signal  import fftconvolve
from   scipy.signal.windows import tukey
from procesamiento import (rmse)


def Hx(x, H):
    Hx = np.real(ifft2(fft2(x, axes=(0,1)) * H, axes=(0,1)))
    
    return Hx

def Hx_t(x, H):
    Hx_t = np.real(ifft2(fft2(x, axes=(0,1)) * np.conjugate(H), axes=(0,1)))
    return Hx_t

def tikhonov(img_degradada, H, lambbda, tau, iter, tolerancia, img_og):
    x = img_degradada.copy()

    errores = []
    costos = []
    x_best, e_best = x.copy(), np.inf

    for i in range(iter):


        residual = Hx(x, H) - img_degradada
        grad = Hx_t(residual, H) + lambbda * x # gradiente
        x_act = x - tau * grad
        func_costo = 0.5*np.sum((Hx(x, H) - img_degradada)**2) + 0.5*lambbda*np.sum(x**2)
        costos.append(func_costo)
        errores.append(rmse(x, img_og))

        if errores[-1] < e_best: 
                                 
            e_best, x_best = errores[-1], x.copy() # recordar el mejor error

        
        if not np.isfinite(costos[-1]) or costos[-1] > 1e12:   # divergence guard
            break

        # para parar se analiza el cambio relativo entre cada iteracion, si supera cierto
        relative_change = np.linalg.norm(x_act - x) / (np.linalg.norm(x) + 1e-12) # para evitar dividir por 0
        if relative_change < tolerancia:
            x = x_act
            break
        x = x_act

    return x, errores, costos, x_best

 



