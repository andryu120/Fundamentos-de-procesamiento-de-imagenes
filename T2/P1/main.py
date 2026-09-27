import numpy as np
import cv2
from creacion_imagen import imagen_ruidosa
from procesamiento import (filtro_gaussiano, mostrar_imagen, rmse)
from creacion_imagen import image

if __name__ == "__main__":

    for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:

        lista = []
        imagen_modificada = filtro_gaussiano(imagen_ruidosa,i)
        error = rmse(imagen_ruidosa, imagen_modificada)

        print(error)

