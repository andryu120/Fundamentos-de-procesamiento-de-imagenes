import io
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import pywt
from scipy import fft, ndimage
from scipy.linalg import hadamard
from PIL import Image, ImageFilter
import cv2


def piramide_laplaciana(imagen, factor=2, umbral=False, a=1, k=1):

    niveles_gauss = []
    lista_umbral = []
    coef = []
    sigma = factor * 0.5
    # crear piramide gaussiana
    for i in range(6):
        if i == 0:
            G_0 = imagen
            niveles_gauss.append(G_0)
            
        else:
            image_blur = cv2.GaussianBlur(niveles_gauss[i-1], (0, 0), sigmaX=sigma)
            h, w = image_blur.shape[:2]
            sub = cv2.resize(image_blur, None, fx=1/factor, fy=1/factor, interpolation=cv2.INTER_LINEAR)
            niveles_gauss.append(sub)
            
    piramide = []

    for i, g in enumerate(niveles_gauss):
        if (i+1) == 6:
            l = niveles_gauss[i]
            piramide.append(l)
            lista_umbral.append(l)
            break
        else:
            if umbral:

                up = cv2.resize(niveles_gauss[i+1], None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
                l = niveles_gauss[i] - up
                t = a * np.max(np.abs(l))
                mascara = np.abs(l) >= t
                d_hat = np.where(mascara, l, 0.0)
                lista_umbral.append(d_hat)
                coef.append(np.mean(mascara) * 100.0) # en porcentajes
                piramide.append(l)
            else:
                up = cv2.resize(niveles_gauss[i+1], None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
                l = k * (niveles_gauss[i] - up)
                piramide.append(l)

    return piramide, lista_umbral, coef

def reconstruccion_laplace(piramide: list, factor=2):
    
    valores = []
    for i in range(len(piramide)):
        indice = len(piramide)-i-1
        if indice == 5:
            valores.append(piramide[indice]) # a;ade la cima
        else:
            up = cv2.resize(valores[-1], None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR) # interpolacion bilineal
            
            g = piramide[indice] + up
            valores.append(g)

    return valores[len(valores)-1] # retorna como el principio de la gaussiana



