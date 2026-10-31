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



def tikhonov(img_degradada, H, lambbda, tau, max_iter, tolerancia, img_og):
    

    rows, cols, ch = img_degradada.shape
    N = rows * cols
    Y  = fft2(img_degradada,      axes=(0, 1))
    Xt = fft2(img_og, axes=(0, 1))
    X = np.zeros_like(Y)

    errores = []
    costos = []
    x_best, e_best = X.copy(), np.inf

    for i in range(max_iter):


        residual = H * X - Y
        grad = np.conj(H) * residual + lambbda * X # gradiente
        X_act = X - tau * grad
        func_costo = (0.5*np.sum(np.abs(residual)**2) + 0.5*lambbda*np.sum(np.abs(X)**2)) / N
        costos.append(func_costo)
        errores.append(np.sqrt(np.sum(np.abs(X - Xt)**2) / (N * N * ch)))

        if errores[-1] < e_best: 
                                 
            e_best, x_best = errores[-1], X.copy() # recordar el mejor error

        
        if not np.isfinite(costos[-1]) or costos[-1] > 1e8:   # divergence guard
            break

        # para parar se analiza el cambio relativo entre cada iteracion, si supera cierto
        relative_change = np.linalg.norm(X_act - X) / (np.linalg.norm(X) + 1e-12) # para evitar dividir por 0
        if relative_change < tolerancia:
            X = X_act
            break
        X = X_act

    a_img = lambda Z: np.real(ifft2(Z, axes=(0, 1)))

    return a_img(X), np.array(costos), np.array(errores), a_img(x_best)

 



