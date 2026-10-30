import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
import os
from   scipy.fft     import fft, fft2, ifft2, fftshift, ifftshift, fftfreq
from   scipy.ndimage import gaussian_filter, convolve, convolve1d
from   scipy.signal  import fftconvolve
from   scipy.signal.windows import tukey
from procesamiento import centrar_psf





def wiener(img_degradada, H, K):
    
    Y = fft2(img_degradada, axes=(0,1))


    F = Y*((np.conjugate(H))/((np.abs(H))**2 + K)) 

    f = ifft2(F, axes=(0,1))

    return np.real(f)




    