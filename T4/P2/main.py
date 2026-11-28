import os
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import (rgb2lab, lab2lch, lch2lab, lab2rgb, lab2xyz)
from   scipy.fft  import fft2, ifft2, fftshift, ifftshift, dctn, idctn
from utils import luminancia, a_pantalla
from procesamiento import por_canal, cargar_rgb
from homorfico import homomorfico_hdr
from durand import durand
from reconstruccion_luminancia import re_lum
from reconstruccion_color import re_color
from graficar import mostrar, elegir_recortes, recortar


PARAMS = {
    "homorfico": "gH=1.5, gL=0.25, D0=40, c=1, pct=[0.5,99]",
    "durand": "sigma_s=2% llado mayor, sigma_r=0.4, contraste=30",
}
EPS_LUM =  1e-4

def to_lch(rgb):
    img_clip = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    return lab2lch(rgb2lab(img_clip))

def metricas_color(img_antes, img_despues, c_min = 5.0):
    lch_antes = to_lch(img_antes)
    lch_despues = to_lch(img_despues)

    c_antes = lch_antes[..., 1]
    c_despues = lch_despues[..., 1]
    h_antes = lch_antes[..., 2]
    h_despues = lch_despues[..., 2]

    diff = np.abs(h_antes - h_despues) % (2.0 * np.pi)
    _diff_angular_min = np.minimum(diff, 2.0 * np.pi - diff)
    delta_h_deg = np.degrees(_diff_angular_min)
    mascara = c_antes > c_min

    if np.any(mascara):
        delta_h_media = float(np.mean(delta_h_deg[mascara]))
        delta_h_mediana = float(np.median(delta_h_deg[mascara]))
        delta_h_p95 = float(np.percentile(delta_h_deg[mascara], 95))
    else:
        delta_h_media = delta_h_mediana = delta_h_p95 = float("nan")

    return {
        "croma_antes_media": float(np.mean(c_antes)),
        "croma_despues_media": float(np.mean(c_despues)),
        "delta_croma_medio": float(np.mean(c_despues - c_antes)),
        "delta_tono_deg_media": delta_h_media,
        "delta_tono_deg_mediana": delta_h_mediana,
        "delta_tono_deg_p95": delta_h_p95,
        "pct_pixeles_tono_validos": float(100.0 * np.mean(mascara)),
    }

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

                    
                    imgs_beta.append(img_vis)
                    titulos_beta.append(f"{metodo} | beta={beta}")
                else:
                    img_rec = re_lum(img_hdr, Y_durand, L_durand, beta=beta)
                    # Clipping solo para visualización segura en [0, 1]
                    img_vis = a_pantalla(img_rec) 

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
            )


def exp3(rutas, beta=1.0, factor=1.5):
    for ruta in rutas:
        img_hdr = cargar_rgb(ruta)
        Y = luminancia(img_hdr)
        ref = a_pantalla(img_hdr / np.percentile(Y, 99))
        cajas = elegir_recortes(ref, Y)

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
                       f"Razón β={beta}", f"Factor ×{factor}"]

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


def exp4(rutas, beta=1.0, factor=1.5):
    resultados = []
    

    for ruta in rutas:
        img = cargar_rgb(ruta)
        Y = luminancia(img)
        ref = a_pantalla(img / np.percentile(Y, 99))

        pipelines = {
            "homorfico": homomorfico_hdr(img),
            "durand": durand(img),
        }

        for metodo, (Y_m, L_m) in pipelines.items():
            img_canal = a_pantalla(por_canal(img, metodo))
            img_lum = a_pantalla(re_lum(img, Y_m, L_m, beta=beta))
            img_col, _ = re_color(img, Y_m, L_m, factor=factor)

            variantes = {
                "por_canal": img_canal,
                f"razon_beta={beta}": img_lum,
                f"croma_x{factor}": img_col,
            }

            for nombre_var, img_out in variantes.items():
                m = metricas_color(ref, img_out, c_min=5.0)
                fila = {"imagen": ruta, "metodo": metodo, "estrategia": nombre_var, **m}
                resultados.append(fila)
                print(
                    f"{ruta} | {metodo:11s} | {nombre_var:16s} | "
                    f"C* antes={m['croma_antes_media']:.2f} -> desp={m['croma_despues_media']:.2f} | "
                    f"delta h med={m['delta_tono_deg_mediana']:.2f}° (p95={m['delta_tono_deg_p95']:.2f}°)"
                )

    return resultados

if __name__ == "__main__":
    rutas = ["memorial.hdr", "belgium.hdr"]
    
    img1 = a_pantalla(cargar_rgb(rutas[0]))
    img2 = a_pantalla(cargar_rgb(rutas[1]))
    mostrar([img1,img2], titulos=["imagen 1 original","imagen 2 original"])
    exp1(rutas, [1, 1.5, 0.6, 0.1]) #toma valores de beta
    exp2(rutas, [1, 1.5, 0.6, 0.1])
    exp3(["memorial.hdr"])
    exp4(["memorial.hdr"])
    
