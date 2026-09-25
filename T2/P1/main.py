import numpy as np
import cv2
from creacion_imagen import imagen_ruidosa
from procesamiento import (filtro_gaussiano, mostrar_imagen)
if __name__ == "__main__":
    for i in [2,4,6,8]:
        imagen_modificada = filtro_gaussiano(imagen_ruidosa,i)
        mostrar_imagen(imagen_modificada)
