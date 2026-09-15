
from procesamiento import (mostrar_imagen, color_saturation, image)
import os
import cv2
import numpy as np

path = os.getcwd()
# puntos a probar
p_aumento_y_disminucion = [(0, 0.0), (120, 4.0), (240, 4.0), (350, 0.0)] 
p_aumento_intervalo = [(10, 4.0), (60, 0.0), (120, 0.0), (240, 0.0), (340, 4.0)] # aumentar rojos
p_combinacion = [(0, 4.0), (60, 0.0), (120, 4.0), (180, 0.0), (240, 4.0), (300, 0.0)] 


puntos = {
    "Aumento y disminucion de dos intervalos" : p_aumento_y_disminucion,
    "Aumento selectivo de un intervalo": p_aumento_intervalo,
    "Combinacion de aumentos y disminuciones" : p_combinacion

}
#interaccion con el usuario



def guardar_imagenes(imagen: tuple, numero_imagen: str , modo: str):
    carpeta_base = f'imagenes_modificadas_{modo.lower()}'
    #directorio final
    ruta_final = os.path.join(os.getcwd(),carpeta_base,f'imagen_{numero_imagen}')
    os.makedirs(ruta_final, exist_ok=True) # esto es para evitar errores con las carpetas
    # ruta final del archivo
    ruta_archivo = os.path.join(ruta_final, f"{imagen[0]}.tif")
    # pasa a bgr
    img_bgr = cv2.cvtColor(imagen[1], cv2.COLOR_RGB2BGR)
    cv2.imwrite(ruta_archivo, img_bgr)
    

if __name__ == "__main__":
    
    print("imagen 1: pajaros")

    
    imagen_original, imagen_cambiada = image("P1_IMG_2402.tif",path)
    for titulo, p in puntos.items():
        imagen_original, imagen_cambiada = image("P1_IMG_2402.tif",path)
        img_cambiada_hsv = color_saturation(imagen_cambiada, p, "hsv")
        img_cambiada_cie = color_saturation(imagen_cambiada, p, "cie")
        guardar_imagenes((titulo, img_cambiada_hsv), "1", "hsv")
        guardar_imagenes((titulo, img_cambiada_cie), "1", "cie")
    
    guardar_imagenes(("imagen_original", imagen_original), "1", 'hsv')
    guardar_imagenes(("imagen_original", imagen_original), "1", 'cie')

    print("imagen 2, paisaje")
    imagen_original, imagen_cambiada = image("imagen.png",path)
    for titulo, p in puntos.items():
        imagen_original, imagen_cambiada = image("imagen.png",path)
        img_cambiada_hsv = color_saturation(imagen_cambiada, p, "hsv")
        img_cambiada_cie = color_saturation(imagen_cambiada, p, "cie")
        guardar_imagenes((titulo, img_cambiada_hsv), "2", "hsv")
        guardar_imagenes((titulo, img_cambiada_cie), "2", "cie")
            
    guardar_imagenes(("imagen_original", imagen_original), "2", 'hsv')
    guardar_imagenes(("imagen_original", imagen_original), "2", 'cie')
        






    