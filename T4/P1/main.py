
import io
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import pywt
from scipy import fft, ndimage
from scipy.linalg import hadamard
from PIL import Image, ImageFilter
import cv2
from procesamiento import (rmse, cargar_img)
from Laplace import (piramide_laplaciana, reconstruccion_laplace)
from Starlet import(starlet, reconstruccion_starlet)
from graficar import (mostrar, mostrar_lista, mostrar_recortes)

def padding(img, n=5):
    m = 2**n
    h, w = img.shape[:2]
    pad_h = (-h) % m
    pad_w = (-w) % m
    Yp = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")
    return Yp, (h, w)

def recortar(Y, shape_orig):
    h, w = shape_orig
    return Y[:h, :w]



def modificar_capas(pesos: list, lista: list, tipo):
    new_lista = lista.copy()
    if tipo == "Laplace":
        last = new_lista.pop(-1)

    for i, peso in enumerate(pesos):
        new_lista[i] = new_lista[i] * peso

    if tipo == "Laplace":
        new_lista.append(last)

    return new_lista


        



if __name__ == "__main__":

    img = cargar_img("cathedral.tif")
    Yp, shape0 = padding(img) 

    # Probar reconstruccion

    piramide, _, _ = piramide_laplaciana(Yp)
    new_image = reconstruccion_laplace(piramide)

    resultado = recortar(new_image, shape0)
    print(rmse(img, resultado))

    mostrar(img, resultado)

    #Mostrar capa de detalles
    mostrar_lista(piramide, titulo="Laplace")

    capas, c, _, _  = starlet(img, 5)
    new_image = reconstruccion_starlet(capas, c)
    print(rmse(img, new_image))
    mostrar(img, new_image)
    mostrar_lista(capas, titulo="starlet")

    #Umbral

    for i in [0.01, 0.2]:

        piramide, lista_umbral, coef = piramide_laplaciana(Yp, umbral=True, a=i)
        
        
        new_image = reconstruccion_laplace(lista_umbral)
        print("Piramide Laplaciana")
        for i in coef:
            print(i)

        resultado = recortar(new_image, shape0)
        mostrar(img, resultado)

        capas, c, lista_umbral, coef  = starlet(img, 5, umbral=True, a=i)
        new_image = reconstruccion_starlet(lista_umbral, c)
        print("Starlet")
        for i in coef:
            print(i)
        
        mostrar(img, new_image)

    
    p_grueso = [1, 1, 1, 2.5, 2.5]
    p_fino = [2.5, 2.5, 1, 1, 1]
    piramide, lista_umbral, coef = piramide_laplaciana(Yp)

    new_piramide = modificar_capas(p_grueso, piramide, "Laplace")
    
    new_image = reconstruccion_laplace(new_piramide)
    

    resultado = recortar(new_image, shape0)
    

    capas, c, lista_umbral, coef  = starlet(img, 5, umbral=True, a=0.1)
    new_capas = modificar_capas(p_grueso, capas, None)
    new_image = reconstruccion_starlet(new_capas, c)

    mostrar_recortes([img, resultado, new_image],
                 ["Original", "Laplace", "Starlet"],
                 ventana=(250, 250, 150, 150))

    
    mostrar(img - resultado, img - new_image)
    
    
        
        

        




