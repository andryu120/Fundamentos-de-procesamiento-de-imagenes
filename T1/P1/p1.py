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


def correcion_m(p: list):
    "Se define un numero menor a entre 0 y 1 como atenuacion, y 1 a 10 como amplificacion, siendo 1 el neutro"
    # Esta funcion corrige los valores otorgados en la lista p
    
    for i in range(len(p)):
        if p[i][1] > 10:
            p[i] = (p[i][0],10.0)
        elif p[i][1] < 0:
            p[i] = (p[i][0],0)

    return p
        

# interpolacion con la expecion ciclica


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
    
    
    for i in range(len(p)-2):
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
    arreglo_temporal[mascara_pixeles_bajos] = arreglo_temporal[mascara_pixeles_bajos] + 360.0
    m_base[mascara_ultimo_punto] = m_final + ((m_inicial-m_final)/(h_inicial+360 - h_final)) *(arreglo_temporal[mascara_ultimo_punto] - h_final)

    return m_base


# transformacion de la saturacion

def transformacion(img: np.ndarray, m_base: np.ndarray):
    # para transformar la matriz original, se va a multiplicar por esta nueva matriz de parametros m
    # Saturacion
    S = img[:, :, 1]

    S_prima = S * m_base
    return S_prima


#actualizacion de la imagen con la nueva saturacion


def hsv_to_rgb(hsv_img: np.ndarray, S: np.ndarray) -> np.ndarray:
    # canales
    H = hsv_img[:, :, 0]
    S = transformacion(hsv_img, S)
    V = hsv_img[:, :, 2]

    
    C = V * S
    H_prime = H / 60.0
    
    
    X = C * (1.0 - np.abs((H_prime % 2.0) - 1.0))
    m = V - C

   
    R1 = np.zeros_like(H)
    G1 = np.zeros_like(H)
    B1 = np.zeros_like(H)


    mask_0 = (0 <= H_prime) & (H_prime < 1)
    mask_1 = (1 <= H_prime) & (H_prime < 2)
    mask_2 = (2 <= H_prime) & (H_prime < 3)
    mask_3 = (3 <= H_prime) & (H_prime < 4)
    mask_4 = (4 <= H_prime) & (H_prime < 5)
    mask_5 = (5 <= H_prime) & (H_prime <= 6)

    R1[mask_0], G1[mask_0] = C[mask_0], X[mask_0]
    R1[mask_1], G1[mask_1] = X[mask_1], C[mask_1]
    G1[mask_2], B1[mask_2] = C[mask_2], X[mask_2]
    G1[mask_3], B1[mask_3] = X[mask_3], C[mask_3]
    R1[mask_4], B1[mask_4] = X[mask_4], C[mask_4]
    R1[mask_5], B1[mask_5] = C[mask_5], X[mask_5]

    R = R1 + m
    G = G1 + m
    B = B1 + m


    rgb_img = np.stack([R, G, B], axis=-1)
    rgb_img = rgb_img * 255.0


    rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)

    return rgb_img


def mostrar_imagen(img: np.ndarray):
    # # Si queremos mostrala
    plt.figure(figsize= (15,8))
    plt.imshow(img)
    plt.show()




