import os
import cv2
import matplotlib.pyplot as plt
import numpy as np




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
    Xbgr = cv2.imread(name)
    imagen = cv2.cvtColor(Xbgr, cv2.COLOR_BGR2RGB) # conversión de BGR a RGB

    return (imagen,imagen.copy())





# usando HSV, hay que procesar la matriz en el espacio hsv, luego en el otro

# codigo sacado de la capsula
def rgb_to_hsv(img: np.ndarray) -> np.ndarray:

  img = img.astype(np.float64)
  img = img/ 255.0

  R = img[:, :, 0]
  G = img[:, :, 1]
  B = img[:, :, 2]

  #Nuevamente se define un epsilon pensando en caso borde de imagen negra
  epsilon = 1e-10

  # Se definen los C y delta
  C_max = np.maximum(np.maximum(R, G), B)
  C_min = np.minimum(np.minimum(R, G), B)
  delta = C_max - C_min

  # V es directamente el valor máximo
  V = C_max

  # S establece la condicion de saturacion
  S = np.where(C_max > epsilon, delta / (V + epsilon), 0.0)

  # Para el cálculo de H se define una matriz a "rellenar"
  H = np.zeros_like(V)

  # De acuerdo con el canal dominante por pixel creamos mascaras booleanas

  mask_r = (C_max == R) * (delta > epsilon)
  mask_g = (C_max == G) * (delta > epsilon)
  mask_b = (C_max == B) * (delta > epsilon)

  # Se define la fórmula correspondiente por color dominante

  H[mask_r] = 60.0 * (((G[mask_r] - B[mask_r]) / delta[mask_r]) % 6)
  H[mask_g] = 60.0 * (((B[mask_g] - R[mask_g]) / delta[mask_g]) + 2)
  H[mask_b]= 60.0 * (((R[mask_b] - G[mask_b]) / delta[mask_b]) + 4)

  #Si algún grado quedo menor a 0 sumamos 360°

  H = np.where(H < 0, H + 360.0, H)

  #Para zonas grises
  H = np.where(delta < epsilon, 0.0, H)

  hsv_img = np.stack([H, S, V], axis=-1)
  return hsv_img

imagen_original, imagen = image("imagen.png", path)

img_hsv = rgb_to_hsv(imagen)

#obtencion de h y s?




# interpolacion con la expecion ciclica


def interpolar(img: np.ndarray, p: list):
    #seleccionamos los valores de h y s de la imagen
   
    H  = img_hsv[:, :, 0]
    S = img_hsv[:, :, 1]
    # interpolacion clasica
    m_base = np.zeros_like(H)
    #ordena de menor a mayor
    p = sorted(p)
    if len(p) < 2:
        raise ValueError("Lista no tiene suficientes puntos")
    
    
    for i in range(p-2):
        h1, m1 = p[i]
        h2, m2 = p[i+1]

        
        mascara = (H >= h1) and (H < h2)
        m_base[mascara] = m1 + ((m2-m1)/(h2-h1)) * (H[mascara]-h1)

    # para el ultimo punto
    



    

    

# transformacion de la saturacion

def transformacion():
    pass


#actualizacion de la imagen con la nueva saturacion

def actualizar_imagen():
    pass

def mostrar_imagen():
    pass



# # Si queremos mostrala
# plt.figure(figsize= (15,8))
# plt.imshow(imagen_hsv)
# plt.show()