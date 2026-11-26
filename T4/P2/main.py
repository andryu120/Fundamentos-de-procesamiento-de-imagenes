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
from graficar import mostrar

def exp1(rutas, betas):
    metodos = ["homomorfico", "durand"]
    for ruta in rutas:
        img_hdr = cargar_rgb(ruta)
        

        for metodo in metodos:
            imgs_beta = []
            titulos_beta = []

            for beta in betas:
                img_rec = re_lum(img_hdr, metodo=metodo, beta=beta,)
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


def exp2(rutas, factor):
    metodos = ["homomorfico", "durand"]
    for ruta in rutas:
        img_hdr = cargar_rgb(ruta)
        

        for metodo in metodos:
            imgs_beta = []
            titulos_beta = []

            for f in factor:
                img_rec = re_color(img_hdr, metodo=metodo, factor=f,)
                # Clipping solo para visualización segura en [0, 1]
                img_vis = a_pantalla(img_rec) 

                #resultados[ruta][(metodo, beta)] = img_rec
                imgs_beta.append(img_vis)
                titulos_beta.append(f"{metodo} | beta={f}")

            mostrar(
                imgs_beta,
                titulos_beta,
                ncols=len(betas),
                size=4,
            )



if __name__ == "__main__":
    rutas = ["memorial.hdr", "belgium.hdr"]
    #exp1(rutas, [1, 0.6, 0.1])
    exp2(rutas, [1, 0.6, 0.1])