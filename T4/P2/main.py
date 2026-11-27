import os
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from   scipy.fft  import fft2, ifft2, fftshift, ifftshift, dctn, idctn
from utils import luminancia, a_pantalla
from procesamiento import por_canal, cargar_rgb
from homorfico import homomorfico_hdr
from durand import durand
from reconstruccion_luminancia import re_lum
from reconstruccion_color import re_color
from graficar import mostrar, elegir_recortes, recortar, mostrar_cajas


PARAMS = {
    "homorfico": "gH=1.5, gL=0.25, D0=40, c=1, pct=[0.5,99]",
    "durand": "sigma_s=2% llado mayor, sigma_r=0.4, contraste=30",
}
EPS_LUM =  1e-4 

def exp1(rutas, betas):
    metodos = ["homorfico", "durand"]
    for ruta in rutas:
        img_hdr = cargar_rgb(ruta)
        Y_durand, L_durand = durand(img_hdr)
        Y_homorfico, L_homorfico = homomorfico_hdr(img_hdr)
        

        for metodo in metodos:
            imgs_beta = []
            titulos_beta = []

            for beta in betas:
                if metodo == "homorfico":

                    img_rec = re_lum(img_hdr, Y_homorfico, L_homorfico, beta=beta)
                    # Clipping solo para visualización segura en [0, 1]
                    img_vis = a_pantalla(img_rec) 

                    #resultados[ruta][(metodo, beta)] = img_rec
                    imgs_beta.append(img_vis)
                    titulos_beta.append(f"{metodo} | beta={beta}")
                else:
                    img_rec = re_lum(img_hdr, Y_durand, L_durand, beta=beta)
                    # Clipping solo para visualización segura en [0, 1]
                    img_vis = a_pantalla(img_rec) 

                    #resultados[ruta][(metodo, beta)] = img_rec
                    imgs_beta.append(img_vis)
                    titulos_beta.append(f"{metodo} | beta={beta}")

            mostrar(
                imgs_beta,
                titulos_beta,
                ncols=len(betas),
                size=4,
            )


def exp2(rutas, factores):
    for ruta in rutas:
        img_hdr = cargar_rgb(ruta)
        # CAMBIO: un solo diccionario en vez de duplicar el bloque if/else por método
        pipelines = {"homorfico": homomorfico_hdr(img_hdr), "durand": durand(img_hdr)}

        for metodo, (Y_m, L_m) in pipelines.items():
            imgs_beta = []
            titulos_beta = []

            for factor in factores:
                # CAMBIO: ya no se descarta el segundo valor: es la fracción fuera de gamut
                img_rec, fuera = re_color(img_hdr, Y_m, L_m, factor=factor)
                imgs_beta.append(img_rec)       # re_color ya entrega sRGB: no aplicar a_pantalla
                titulos_beta.append(f"factor={factor} | fuera={fuera:.1%}")

            # CAMBIO: título general con imagen, método y parámetros
            mostrar(
                imgs_beta,
                titulos_beta,
                ncols=len(factores),
                size=4,
                suptitulo=f"{ruta} | {metodo} [{PARAMS[metodo]}] | croma CIELCh",
            )


def exp3(rutas, beta=1.0, factor=1.5):
    for ruta in rutas:
        img_hdr = cargar_rgb(ruta)
        Y = luminancia(img_hdr)
        ref = a_pantalla(img_hdr / np.percentile(Y, 99))
        cajas = elegir_recortes(ref, Y)
        mostrar_cajas(ref, cajas, f"{ruta}: recortes")

        # CAMBIO: un solo bloque para ambos métodos (antes el homomórfico se graficaba dos
        # veces por la indentación del segundo bloque)
        pipelines = {"homorfico": homomorfico_hdr(img_hdr), "durand": durand(img_hdr)}

        for metodo, (Y_m, L_m) in pipelines.items():
            rec_lum = re_lum(img_hdr, Y_m, L_m, beta)
            rec_color, fuera_color = re_color(img_hdr, Y_m, L_m, factor)
            # CAMBIO: fracción de píxeles con algún canal fuera de [0,1] ANTES del clip
            fuera_lum = np.mean(np.any((rec_lum < 0) | (rec_lum > 1), axis=-1))
            print(f"{ruta} | {metodo}: fuera de rango razón(beta={beta}) = {fuera_lum:.2%}, "
                  f"croma(x{factor}) = {fuera_color:.2%}")

            imgs = [
                ref,
                a_pantalla(L_m),
                a_pantalla(por_canal(img_hdr, metodo)),
                a_pantalla(rec_lum),
                rec_color,                      # ya está en sRGB
            ]
            titulos = ["Referencia", "Luminancia comprimida", "Por canal RGB",
                       f"Razón β={beta}", f"Croma ×{factor}"]

            regiones = {"completa": None, "recorte color": cajas["color"],
                        "recorte iluminación": cajas["ilum"]}
            for reg, caja in regiones.items():
                # CAMBIO: título general con imagen, método, región y parámetros
                mostrar([recortar(im, caja) for im in imgs],
                        titulos,
                        cmap=["gray" if i == 1 else None for i in range(5)],
                        ncols=5, size=3.5, vmin=0, vmax=1,
                        suptitulo=f"{ruta} | {metodo} [{PARAMS[metodo]}] | {reg} | "
                                  f"eps={EPS_LUM}")




if __name__ == "__main__":
    rutas = ["memorial.hdr", "belgium.hdr"]
    #exp1(rutas, [1, 0.6, 0.1])
    #exp2(rutas, [1, 0.6, 0.1])
    exp3(["memorial.hdr"])