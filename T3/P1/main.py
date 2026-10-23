import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt

from procesamiento import (image, fourier)

def espectro_log(F):
    # log(1 + |F|) permite ver a la vez la componente continua y las frecuencias altas
    return np.log1p(np.abs(F))

def mostrar_imagen(img: np.ndarray, titulo: str):
    import matplotlib.pyplot as plt
    # # Si queremos mostrala
    
    plt.figure(figsize=(15,8))
    plt.imshow(img)
    plt.title(f'{titulo}')
    plt.show()

if __name__ == "__main__":
    imagen = image("cameraman.jpg")
    fft = fourier(imagen)
    espectro = espectro_log(fft)
    mostrar_imagen(espectro, "nigga ")
