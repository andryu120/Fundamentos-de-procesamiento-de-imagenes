import os
import numpy as np
import matplotlib as plt
import cv2


from espacios_color import (rgb_to_hsv,rgb_to_lch,lch_to_rgb,hsv_to_rgb)


# Source - https://stackoverflow.com/a/1724723
# Posted by Nadia Alramli, modified by community. See post 'Timeline' for change history
# Retrieved 2026-08-07, License - CC BY-SA 2.5
path = os.getcwd()
#Importar una imagen
def image(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            ruta = os.path.join(root, name)
    name = ruta
    Xbgr = cv2.imread(name,cv2.IMREAD_UNCHANGED)
    imagen = cv2.cvtColor(Xbgr, cv2.COLOR_BGR2RGB) # conversion de BGR a RGB

    return (imagen,imagen.copy().astype(np.float32))

def correcion_m(p: list):
    "Se define un numero menor a entre 0 y 1 como atenuacion, y 1 a 10 como amplificacion, siendo 1 el neutro"
    # Esta funcion corrige los valores otorgados en la lista p
    
    for i in range(len(p)):
        if p[i][1] > 10:
            p[i] = (p[i][0],10.0)
        elif p[i][1] < 0:
            p[i] = (p[i][0],0)

    return p


# interpolacion con la excepcion ciclica


def interpolar(img: np.ndarray, p: list):
    #seleccionamos los valores de h y s de la imagen
    # p es una lista con tuples
    H  = img[:, :, 0]
    # interpolacion clasica

    #copia de la matriz*** (matriz del mismo tama;o con ceros)
    m_base = np.zeros_like(H)
    #ordena de menor a mayor
    p = sorted(p)
    if len(p) < 2:
        raise ValueError("Lista no tiene suficientes puntos")
    
    
    for i in range(len(p)-1):
        h1, m1 = p[i]
        h2, m2 = p[i+1]

        # booleano para seleccionar los puntos
        mascara = (H >= h1) & (H < h2)
        m_base[mascara] = m1 + ((m2-m1)/(h2-h1)) * (H[mascara]-h1)

    #interpolacion ciclica

    h_final, m_final = p[-1]
    h_inicial, m_inicial = p[0]

    mascara_ultimo_punto = (H >= h_final) | (H <h_inicial)

    #ajuste temporal

    arreglo_temporal = np.copy(H)

    mascara_pixeles_bajos = (H < h_inicial)
    arreglo_temporal[mascara_pixeles_bajos] = arreglo_temporal[mascara_pixeles_bajos] + 360.0 # suma a los pixeles menores 
    
    m_base[mascara_ultimo_punto] = m_final + ((m_inicial-m_final)/(h_inicial+360 - h_final)) *(arreglo_temporal[mascara_ultimo_punto] - h_final)

    return m_base


# transformacion de la saturacion

def transformacion_hsv(img: np.ndarray, m_base: np.ndarray):
    # para transformar la matriz original, se va a multiplicar por esta nueva matriz de parametros m
    # Saturacion
    
    S = img[:, :, 1]

    S_prima = S * m_base
    #S_prima = np.clip(S_prima, 0.0, 1.0) 
    return S_prima

def transformacion_lcab(img: np.ndarray, m_base: np.ndarray):
    
    
    C = img[:, :, 1]
    C_prima = C * m_base
    return C_prima


def mostrar_imagen(img: np.ndarray):
    import matplotlib.pyplot as plt
    # # Si queremos mostrala
    plt.figure(figsize=(15,8))
    plt.imshow(img)
    plt.show()


def color_saturation(imagen:  np.ndarray, p: list, modo: str):
    if modo in ["HSV", "hsv"]:
        
       
        imagen_hsv = rgb_to_hsv(imagen)
        p = correcion_m(p)
        M = interpolar(imagen_hsv, p)
        S_prima = transformacion_hsv(imagen_hsv,M)
        imagen_rgb = hsv_to_rgb(imagen_hsv, S_prima)
        return imagen_rgb

    elif modo in ["CIE", "cie"]:
        
        
        imagen_lch = rgb_to_lch(imagen)
        p = correcion_m(p)
        M = interpolar(imagen_lch, p)
        C_prima = transformacion_lcab(imagen_lch,M)
        imagen_rgb = lch_to_rgb(imagen_lch, C_prima)
        return imagen_rgb
    else:
        raise ValueError("Modo incorrecto, ingrese de nuevo el modo")

def mostrar_pixeles(imagen: np.ndarray, p: list, y: int, x: int, modo: str):
    if modo in ["HSV","hsv"]:

        print(f"Coordenadas del pixel (Y={y}, X={x}) en modo {modo.upper()}")

        R_orig, G_orig, B_orig = imagen[y,x]
        print(f"Coordenadas RGB Original: (R={R_orig}, G={G_orig}, B={B_orig})")


        img_hsv = rgb_to_hsv(imagen)
        h, s, v = img_hsv[y,x]
        print(f"Coordenadas HSV: (H={h}, s={s}, v={v})")


        p = correcion_m(p)
        m = interpolar(img_hsv, p)
        print(f"m(h) (interpolado): {m[y,x]}")
        

        S_prima = transformacion_hsv(img_hsv, m)
        print(f"Resultado de g(m): {S_prima[y,x]}")

        img_final = hsv_to_rgb(img_hsv, S_prima)

        R_final, G_final , B_final = img_final[y,x]
        print(f"Coordenadas RGB Finales: (R={R_final}, G={G_final}, B={B_final})\n")

    else:
        print(f"Coordenadas del pixel (Y={y}, X={x}) en modo {modo.upper()}")
        
        R_orig, G_orig, B_orig = imagen[y,x]
        print(f"Coordenadas RGB Original: (R={R_orig}, G={G_orig}, B={B_orig})")


        img_cie = rgb_to_lch(imagen)
        l, c, h = img_cie[y,x]
        print(f"Coordenadas CIE: (L={l}, c={c}, H={h})")


        p = correcion_m(p)
        m = interpolar(img_cie, p)
        print(f"m(h) (interpolado): {m[y,x]}")
        

        C_prima = transformacion_lcab(img_cie, m)
        print(f"Resultado de g(m): {C_prima[y,x]}")

        img_final = lch_to_rgb(img_cie, C_prima)

        R_final, G_final , B_final = img_final[y,x]
        print(f"Coordenadas RGB Finales: (R={R_final}, G={G_final}, B={B_final})\n")



