import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
import os
from   scipy.fft     import fft, fft2, ifft2, fftshift, ifftshift, fftfreq
from   scipy.ndimage import gaussian_filter, convolve, convolve1d
from   scipy.signal  import fftconvolve
from   scipy.signal.windows import tukey



#Importar una imagen
def image(name):
    path = os.getcwd()
    for root, dirs, files in os.walk(path):
        if name in files:
            ruta = os.path.join(root, name)
    name = ruta
    imagen = cv2.imread(name, cv2.IMREAD_GRAYSCALE) # leer en escalas de grises
    return imagen.astype(float)

def espectro_log(F):
    # log(1 + |F|) permite ver a la vez la componente continua y las frecuencias altas
    db = np.log1p(np.abs(F))
    return db

def mostrar_imagen(img: np.ndarray, titulo: str):
    # # Si queremos mostrala
    
    plt.figure(figsize=(15,8))
    plt.imshow(img, cmap= 'gray')
    plt.title(f'{titulo}')
    plt.show()

def snap_to_peak(mag, col, row, win=6):
    r0, c0 = int(round(row)), int(round(col))
    r1, r2 = max(r0 - win, 0), min(r0 + win + 1, mag.shape[0])
    c1, c2 = max(c0 - win, 0), min(c0 + win + 1, mag.shape[1])
    dr, dc = np.unravel_index(np.argmax(mag[r1:r2, c1:c2]), (r2 - r1, c2 - c1))
    return c1 + dc, r1 + dr                      

def pick_points(imagen, pad, n=2):

    img_padded = np.pad(imagen, pad, mode='edge')
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img_padded))))
    plt.figure(figsize=(10, 10))
    plt.imshow(mag, cmap='gray')
    plt.title(f'Click {n} noise peaks (one per symmetric pair)')
    clicks = [snap_to_peak(mag, c, r) for c, r in plt.ginput(n, timeout=0)]    
    plt.close()
    R , C = mag.shape

    return [((c - C//2) / C, (r - R//2) / R) for c, r in clicks]
    
    

def notch_filter(imagen,puntos, radio, mode, pad, tipo):
    "Recibe una lista [(col_k, row_k)], el modo y la cantidad de zeros para el padding, ademas del tipo de filtro. Retorna la imagen modificada, el espectro original, el espectro modificado"

    rows, cols = imagen.shape

    #aplicar o no padding

    if mode == 'zeros':
        img_padded = np.pad(imagen, pad, mode='constant', constant_values=0)
    elif mode in ('reflect', 'edge'):
        img_padded = np.pad(imagen, pad, mode=mode)
    else:
        pad = 0
        img_padded = imagen.copy()
        

    
    # Transformada de Fourier 
    f_transform = np.fft.fft2(img_padded)
    f_shift = np.fft.fftshift(f_transform)

    # Coordenadas y matriz de distancia respecto al centro
    padded_rows, padded_cols = img_padded.shape
    crow, ccol = padded_rows // 2, padded_cols // 2

    mask_notch = np.ones((padded_rows, padded_cols))

    U, V = np.indices((padded_rows, padded_cols))

    for i in range(len(puntos)):
        fx, fy = puntos[i]
        col_k = ccol + fx * padded_cols
        row_k = crow + fy * padded_rows
        col_k_ , row_k_ = (2*ccol - col_k, 2*crow - row_k)

        Dk = ((U - row_k)**2 + (V - col_k)**2)
        D_menos_k = ((U - row_k_)**2 + (V - col_k_)**2)

        if tipo == 'ideal':

            mask_notch[np.sqrt(Dk) <= radio] = 0
            mask_notch[np.sqrt(D_menos_k) <= radio] = 0
        else:
            notch_k = 1 - np.exp(-Dk / (2 * radio**2))
            notch_menos_k = 1 - np.exp(-D_menos_k / (2 * radio**2))
            mask_notch *= (notch_k * notch_menos_k)
    # Para la pregunta guia
    if i == 0:
            idx_r, idx_c = int(row_k), int(col_k)
            valor_original = f_shift[idx_r, idx_c]
            valor_H = mask_notch[idx_r, idx_c]
            print(f"F(u,v): {valor_original}")
            print(f"H(u,v): {valor_H}")
            print(f"Resultado: {valor_original * valor_H}")
            
    new_fft = np.real(np.fft.ifft2(np.fft.ifftshift(f_shift * mask_notch)))[pad:pad+rows, pad:pad+cols]

    return new_fft, f_shift, f_shift * mask_notch, mask_notch
