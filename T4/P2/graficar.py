import os
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from   scipy.fft  import fft2, ifft2, fftshift, ifftshift, dctn, idctn




def mostrar(imgs, titulos, cmap='gray', ncols=None, size=4, vmin=None, vmax=None, marco=False):
    # Muestra una lista de imágenes lado a lado; cmap puede ser uno o una lista
    n = len(imgs)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    cmaps = cmap if isinstance(cmap, list) else [cmap] * n

    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, img, titulo, cm in zip(axes, imgs, titulos, cmaps):
        ax.imshow(img, cmap=cm, vmin=vmin, vmax=vmax)
        ax.set_title(titulo)
    for ax in axes:
        if marco:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()