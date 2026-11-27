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
    metodos = ["homorfico", "durand"]
    for ruta in rutas:
        img_hdr = cargar_rgb(ruta)
        Y_durand, L_durand = durand(img_hdr)
        Y_homorfico, L_homorfico = homomorfico_hdr(img_hdr)


        for metodo in metodos:
            imgs_beta = []
            titulos_beta = []

            for factor in factores:
                if metodo == "homorfico":
                    
                    img_rec, _ = re_color(img_hdr, Y_homorfico, L_homorfico, factor=factor)
                    # Clipping solo para visualización segura en [0, 1]
                    #img_vis = a_pantalla(img_rec) 

                    #resultados[ruta][(metodo, beta)] = img_rec
                    imgs_beta.append(img_rec)
                    titulos_beta.append(f"{metodo} | factor={factor}")
                else:
                    img_rec, _ = re_color(img_hdr, Y_durand, L_durand, factor=factor)
                    # Clipping solo para visualización segura en [0, 1]
                    #img_vis = a_pantalla(img_rec) 

                    #resultados[ruta][(metodo, beta)] = img_rec
                    imgs_beta.append(img_rec)
                    titulos_beta.append(f"{metodo} | factor={factor}")

            mostrar(
                    imgs_beta,
                    titulos_beta,
                    ncols=len(factores),
                    size=4,
                )

def exp3(rutas,beta=1.0, factor=1.5):
    metodos = ["homorfico", "durand"]
    for ruta in rutas:
        img_hdr = cargar_rgb(ruta)
        Y_durand, L_durand = durand(img_hdr)
        Y_homorfico, L_homorfico = homomorfico_hdr(img_hdr)
        Y = luminancia(img_hdr)
        ref = a_pantalla(img_hdr / np.percentile(Y, 99))    
        cajas = elegir_recortes(ref, Y)
        mostrar_cajas(ref, cajas, f"{ruta}: recortes")
        for metodo in metodos:
            if metodo == "homorfico":

                imgs = [
                    ref,
                    a_pantalla(L_homorfico),                      
                    a_pantalla(por_canal(img_hdr, metodo)),       
                    a_pantalla(re_lum(img_hdr,Y_homorfico, L_homorfico, beta)),    
                    re_color(img_hdr, Y_homorfico, L_homorfico, factor)[0],      
                ]
                titulos = ["Referencia", "Luminancia comprimida", "Por canal RGB",
                        f"Razón β={beta}", f"Croma ×{factor}"]

                regiones = {"completa": None, "recorte color": cajas["color"],
                            "recorte iluminación": cajas["ilum"]}
                for reg, caja in regiones.items():
                    mostrar([recortar(im, caja) for im in imgs],
                            [f"{t}" for t in titulos],
                            cmap=["gray" if i == 1 else None for i in range(5)],
                            ncols=5, size=3.5, vmin=0, vmax=1)
            else:
                imgs = [
                    ref,
                    a_pantalla(L_durand),                     
                    a_pantalla(por_canal(img_hdr, metodo)),      
                    a_pantalla(re_lum(img_hdr,Y_durand, L_durand, beta)),   
                    re_color(img_hdr, Y_durand, L_durand, factor)[0],       
                ]
            titulos = ["Referencia", "Luminancia comprimida", "Por canal RGB",
                    f"Razón β={beta}", f"Croma ×{factor}"]

            regiones = {"completa": None, "recorte color": cajas["color"],
                        "recorte iluminación": cajas["ilum"]}
            for reg, caja in regiones.items():
                mostrar([recortar(im, caja) for im in imgs],
                        [f"{t}" for t in titulos],
                        cmap=["gray" if i == 1 else None for i in range(5)],
                        ncols=5, size=3.5, vmin=0, vmax=1)



if __name__ == "__main__":
    rutas = ["memorial.hdr", "belgium.hdr"]
    #exp1(rutas, [1, 0.6, 0.1])
    #exp2(rutas, [1, 0.6, 0.1])
    exp3(rutas)