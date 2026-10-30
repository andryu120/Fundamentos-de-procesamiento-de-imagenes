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
    imagen = cv2.imread(name, cv2.IMREAD_COLOR_RGB)
    return imagen.astype(float)

def espectro_log(F):
    # log(1 + |F|) permite ver a la vez la componente continua y las frecuencias altas
    db = np.log1p(np.abs(F))
    return db

def mostrar_imagen(img: np.ndarray, titulo: str):
    # # Si queremos mostrala
    img = np.clip(img, 0.0, 1.0)
    
    plt.figure(figsize=(15,8))
    plt.imshow(img)
    plt.title(f'{titulo}')
    plt.show()


def rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))


def centrar_psf(psf, img):
    rows, cols, channels = img.shape
    if psf.ndim == 2:
        psf = psf[:, :, None]

    pr, pc = psf.shape[:2]
    psf_padded = np.zeros((rows, cols, psf.shape[2])) # padding para que tenga el mismo tama;o de la imagen og
    psf_padded[:pr, :pc] = psf
    psf_padded = np.roll(psf_padded, (-(pr//2), -(pc//2)), axis=(0, 1))

    
    return fft2(psf_padded, axes=(0, 1))

def degradar(img_original, psf, sigma, seed = 0):
    rng = np.random.default_rng(seed)
    H = centrar_psf(psf, img_original)

    y = np.real(ifft2(fft2(img_original, axes=(0, 1)) * H, axes=(0, 1))) # convolucion circular
    return y + rng.normal(0, sigma, img_original.shape), H # a;adir ruido


