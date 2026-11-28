import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle




def mostrar(img1, img2, titulos=("Imagen 1", "Imagen 2"), gamma=False,
            rango=(0, 1), cmap="gray"):
    def _vis(Y):
        Y = np.clip(Y, *rango)          # <- antes: np.clip(Y, 0, 1)
        if gamma:
            Y = np.where(Y <= 0.0031308, Y * 12.92, 1.055 * Y ** (1 / 2.4) - 0.055)
        return Y

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for a, im, t in zip(ax, (img1, img2), titulos):
        a.imshow(_vis(im), cmap=cmap, vmin=rango[0], vmax=rango[1])  # <- antes: vmin=0, vmax=1
        a.set_title(t)
        a.axis("off")
    plt.tight_layout()
    plt.show()





def mostrar_lista(imgs, n=None, ncols=3, titulo="Nivel", gamma=False,
                  figsize_por_img=4, rango=(0, 1), titulos=None):
    def _vis(Y, r):
        Y = np.clip(Y, *r)              # <- antes: np.clip(Y, 0, 1)
        if gamma:
            Y = np.where(Y <= 0.0031308, Y * 12.92, 1.055 * Y ** (1 / 2.4) - 0.055)
        return Y

    imgs = imgs[:n] if n is not None else imgs
    n = len(imgs)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    fig, ax = plt.subplots(nrows, ncols,
                           figsize=(figsize_por_img * ncols, figsize_por_img * nrows),
                           squeeze=False)
    for i, a in enumerate(ax.ravel()):
        if i < n:
            r = rango[i] if isinstance(rango[0], (tuple, list)) else rango   # <- nuevo
            t = titulos[i] if titulos is not None else f"{titulo} {i}"       # <- nuevo
            h, w = imgs[i].shape[:2]
            a.imshow(_vis(imgs[i], r), cmap="gray", vmin=r[0], vmax=r[1])
            a.set_title(f"{t}\n{h} × {w}")
        a.axis("off")
    plt.tight_layout()
    plt.show()


def mostrar_recortes(imagenes, titulos, ventana, rango=(0, 1), figsize_por_img=4,
                     color_caja="red"):
    """
    Fila superior: imágenes completas con la ventana marcada.
    Fila inferior: recorte ampliado (interpolación 'nearest', se ven los píxeles).
    Misma ventana y mismo rango de visualización para todas.
 
    imagenes : lista de arrays 2D del MISMO tamaño (recortar el padding antes)
    titulos  : lista de strings (incluya parámetros: alpha, ganancias, ...)
    ventana  : (y0, x0, alto, ancho) en píxeles
    rango    : (vmin, vmax); para capas con signo use (-s, s)
    """
    y0, x0, h, w = ventana
    vmin, vmax = rango
    n = len(imagenes)
 
    fig, ax = plt.subplots(2, n, figsize=(figsize_por_img * n, figsize_por_img * 2),
                           squeeze=False)
    for j, (im, t) in enumerate(zip(imagenes, titulos)):
        im_v = np.clip(im, vmin, vmax)
        ax[0, j].imshow(im_v, cmap="gray", vmin=vmin, vmax=vmax)
        ax[0, j].add_patch(Rectangle((x0, y0), w, h, fill=False,
                                     edgecolor=color_caja, linewidth=1.5))
        ax[0, j].set_title(t, fontsize=9)
        ax[0, j].axis("off")
 
        ax[1, j].imshow(im_v[y0:y0 + h, x0:x0 + w], cmap="gray",
                        vmin=vmin, vmax=vmax, interpolation="nearest")
        ax[1, j].set_title(f"Recorte [{y0}:{y0 + h}, {x0}:{x0 + w}]", fontsize=9)
        ax[1, j].axis("off")
    plt.tight_layout()
    plt.show()