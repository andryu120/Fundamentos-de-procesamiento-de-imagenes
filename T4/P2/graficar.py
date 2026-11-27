import os
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from   scipy.fft  import fft2, ifft2, fftshift, ifftshift, dctn, idctn
from scipy.ndimage import uniform_filter
from skimage.color import rgb2lab, lab2lch
import matplotlib.patches as patches



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







def elegir_recortes(ref, Y, frac=0.2):
    """
    Devuelve dos ventanas (y0, y1, x0, x1):
      'color': mayor croma medio en la referencia sRGB
      'ilum' : mayor gradiente medio de log10(Y) (transición fuerte de iluminación)
    """
    H, W = Y.shape
    h, w = int(H * frac), int(W * frac)
    C = lab2lch(rgb2lab(ref))[..., 1]
    logY = cv2.GaussianBlur(np.log10(Y + 1e-6).astype(np.float32), (0, 0), 2)
    gy, gx = np.gradient(logY)
    mapas = {"color": uniform_filter(C, (h, w)),
             "ilum": uniform_filter(np.hypot(gx, gy), (h, w))}
    valido = np.zeros((H, W), bool)
    valido[h // 2:H - h // 2, w // 2:W - w // 2] = True
    cajas = {}
    for k, M in mapas.items():
        y, x = np.unravel_index(np.argmax(np.where(valido, M, -np.inf)), M.shape)
        y0, x0 = y - h // 2, x - w // 2
        cajas[k] = (y0, y0 + h, x0, x0 + w)
    return cajas


def recortar(img, caja):
    if caja is None:
        return img
    y0, y1, x0, x1 = caja
    return img[y0:y1, x0:x1]


def mostrar_cajas(img, cajas, titulo="Recortes elegidos"):
    # Imagen completa con los rectángulos marcados (útil para el informe)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    for nombre, (y0, y1, x0, x1) in cajas.items():
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       fill=False, edgecolor="red", linewidth=2))
        ax.text(x0, y0 - 5, nombre, color="red", fontsize=9)
    ax.set_title(titulo)
    ax.axis("off")
    plt.tight_layout()
    plt.show()