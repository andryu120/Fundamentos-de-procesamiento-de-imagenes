from skimage import io, color
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import fft, ndimage as ndi
from scipy.linalg import hadamard
import cv2

def rmse(referencia, aproximacion):
    return np.sqrt(np.mean((referencia - aproximacion) ** 2))



def cargar_img(path):
    """
    Carga una imagen y devuelve la iluminancia normalizada
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    img = img[..., :3].astype(np.float32)
    rgb = img[..., ::-1]

    Y = color.rgb2gray(rgb).astype(np.float32)
    Y = Y / Y.max() # normalizar

    # comprimir por el metodo gamma
    gamma = 2.2
    Yc = Y ** (1.0 / gamma)
    Yc = Yc / Yc.max() # normalizar otra vez
    return Yc.astype(np.float32)

    
