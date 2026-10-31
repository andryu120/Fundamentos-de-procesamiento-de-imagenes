import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt


from procesamiento import (mostrar_imagen, image, degradar, rmse, centrar_psf)
from Wiener import wiener
from Tikhonov import tikhonov

from scipy.ndimage import gaussian_filter

# Source - https://stackoverflow.com/a/33639608
# Posted by letmaik
# Retrieved 2026-10-30, License - CC BY-SA 3.0

import rawpy


def graficar_error(img_original, psf):
    Ks = np.logspace(-6, 0, 40) # el espacio de K
    mejor_k = {}
    sigmas = [0.01, 0.05]
    plt.figure()
    for s in sigmas:
        y, H = degradar(img_original, psf, s)
        errs = [rmse(wiener(y, H, K), img_original) for K in Ks]
        mejor_k[s] = Ks[int(np.argmin(errs))]
        plt.loglog(Ks, errs, label=f"sigma={s}  (mejor K={mejor_k[s]:.1e})")
    plt.xlabel("K"); plt.ylabel("RMSE"); plt.legend(); plt.show()
    return mejor_k

def recortes(img_original, img_modificada):
    s = 150
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    ax[0].imshow(np.clip(img_original[1090-s:1090+s, 3350-s:3350+s], 0, 1)); ax[0].set_title("Original")
    ax[1].imshow(np.clip(img_modificada[1090-s:1090+s, 3350-s:3350+s], 0, 1));  ax[1].set_title("Modificada")
    plt.show()

def comparar(x, y, cerrada, iterativa, titulo, zoom=(200, 200, 100)):
    y0, x0, s = zoom                                   
    imgs = [("Ideal", x), ("Degradada", y),
            (f"Cerrada\nRMSE={rmse(cerrada,x):.4f}", cerrada),
            (f"Iterativa final\nRMSE={rmse(iterativa,x):.4f}", iterativa)]
    fig, ax = plt.subplots(2, 4, figsize=(18, 9))
    for j, (nombre, im) in enumerate(imgs):
        im = np.clip(im, 0, 1)
        ax[0, j].imshow(im, vmin=0, vmax=1); ax[0, j].set_title(nombre)
        ax[1, j].imshow(im[y0-s:y0+s, x0-s:x0+s], vmin=0, vmax=1)
        ax[0, j].axis("off"); ax[1, j].axis("off")
        
        ax[0, j].add_patch(plt.Rectangle((x0-s, y0-s), 2*s, 2*s, fill=False, ec="yellow", lw=1))
    plt.suptitle(titulo); plt.tight_layout(); plt.show()





if __name__ == "__main__":
    raw = rawpy.imread("P1.CR2")
    # normalizar
    rgb = raw.postprocess().astype(np.float64) / 255.0

    
    psf = image("psf.png")
    psf = psf / psf.sum(axis=(0, 1), keepdims=True)

    #Parte A 

    # 1

    # graficar_error(rgb, psf)

    mejor_k = {0.01: 7.0e-03,
                 0.05: 4.1e-02} # sigma: K
    
    for sigma, K in mejor_k.items():
        y, H = degradar(rgb, psf, sigma)
        f = wiener(y, H, K)
        recortes(rgb, f)
        mostrar_imagen(f, "modificada")

    2
    psf_erronea = gaussian_filter(psf, sigma=(0.8, 0.8, 0) if psf.ndim == 3 else 0.5)
    psf_erronea /= psf_erronea.sum(axis=(0, 1), keepdims=True)

    

    # for sigma, K in mejor_k.items():
    #         if sigma == 0.05:
    #              break
    #         y, H = degradar(rgb, psf, sigma)             
    #         H_erronea = centrar_psf(psf_erronea, rgb)        

    #         f = wiener(y, H, K)
    #         f_erronea = wiener(y, H_erronea, K)
    #         print(rmse(f, rgb), rmse(f_erronea, rgb))
    #         recortes(rgb, f_erronea)


    # Parte B

    # 3

    # # Recortar la imagen para facilitar el computo
    # x = rgb[1500:2012, 1700:2212]
    # y, H = degradar(x, psf, 0.01)
    # Ks = np.logspace(-6, 0, 40)
    # K = Ks[np.argmin([rmse(wiener(y, H, k), x) for k in Ks])]
    

    # resultados = {}
    # sigma = 0.01                             
    # y, H = degradar(x, psf, sigma)

    # for lam in [0, K, K/10, 10*K]:
    #     for tau in [1.0 ,1.8/(1+lam), 2.1/(1+lam)]:
    #         xk, c, e, _ = tikhonov(y, H, lam, tau, 2000, 0, x)
    #         resultados[(lam, tau)] = (xk, c, e)

    # # graficar costo vs rmse

    # lams = sorted({lam for lam, _ in resultados})
    # estilos = ['-', '--', ':']
    # fig, ax = plt.subplots(2, len(lams), figsize=(18, 7), sharey='row')

    # for j, lam in enumerate(lams):
    #     taus = sorted(t for l, t in resultados if l == lam)
    #     for t, ls in zip(taus, estilos):
    #         xk, c, e = resultados[(lam, t)]
    #         ax[0, j].semilogy(c, ls, label=f"τ={t:.2f}")
    #         ax[1, j].semilogy(e, ls)
    #     ax[0, j].set_title(f"λ={lam:.1e}")
    #     ax[1, j].set_xlabel("iteración")
    #     ax[0, j].legend(fontsize=8)

    # ax[0, 0].set_ylabel("Costo J(x_k)")
    # ax[1, 0].set_ylabel("RMSE(x_k, x)")

    
    # ax[0, 0].set_ylim(10, 1e4)
    
    # ax[1, 0].set_ylim(1e-2, 1e1)

    # plt.suptitle(f"σ={sigma}, crop 512×512, K*={K:.1e}")
    # plt.tight_layout(); plt.show()
    # # ver que pasa con la tolerancia

    # # for tol in [1e-3, 1e-4, 1e-5, 1e-6, 0]:
    # #     xk, c, e, _ = tikhonov(y, H, K, 1.0, 5000, tol, x)
    # #     print(f"tol={tol:.0e}  iters={len(e)}  RMSE={rmse(xk, x):.5f}")

    # # 4

    # # λ = K
    # x_wk = wiener(y, H, K)                                   
    # x_tk, c1, e1,  x_tk_best = tikhonov(y, H, K, 1.0, 6000, 1e-9, x)

    # # λ = 0
    # x_inv = wiener(y, H, 1e-12)                                  
    # x_t0, c0, e0, x_t0_best = tikhonov(y, H, 0.0, 1.0, 3000, 0, x)

    # print(f"{'caso':<12}{'rmse cerrada':>14}{'rmse iter. optima':>20}{'iter optima':>13}")
    # print(f"{'λ = K*':<12}{rmse(x_wk,x):14.5f}{rmse(x_tk_best,x):20.5f}{e1.argmin():13d}")
    # print(f"{'λ = 0':<12}{rmse(x_inv,x):14.5f}{rmse(x_t0_best, x):20.5f}{e0.argmin():13d}")
    # print("error relativo entre ambos casos: ", np.linalg.norm(x_tk-x_wk)/np.linalg.norm(x_wk))

    # comparar(x, y, x_wk,  x_tk, f"λ = K* = {K:.1e}, σ={sigma}, τ=1.0, crop 512×512")
    # comparar(x, y, x_inv, x_t0_best, f"λ = 0, σ={sigma}, τ=1.0, parada en iter {e0.argmin()}, crop 512×512")


    



            



    

     



    
    