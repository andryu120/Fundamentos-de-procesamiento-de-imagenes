import os
import cv2
import matplotlib.pyplot as plt
import numpy as np



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


# inversa de la funcion anterior
def hsv_to_rgb(hsv_img: np.ndarray, S_prima: np.ndarray) -> np.ndarray:
    # canales
    H = hsv_img[:, :, 0]
    S = S_prima
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


def rgb_to_lab(img: np.ndarray) -> np.ndarray:

    #Los parámetros X_n, Y_n, Z_n pueden variar
    _X_n = 1.0
    _Y_n = 1.0
    _Z_n = 1.0

    xyz = rgb_to_xyz(img)

    xr = xyz[:, :, 0] / _X_n
    yr = xyz[:, :, 1] / _Y_n
    zr = xyz[:, :, 2] / _Z_n

    delta = 6.0 / 29.0

    def f(t):
        return np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4.0 / 29.0)

    fx, fy, fz = f(xr), f(yr), f(zr)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    lab_img = np.stack([L, a, b], axis=-1)
    return lab_img


def rgb_to_lch(img: np.ndarray) -> np.ndarray:

    lab = rgb_to_lab(img)

    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    C = np.sqrt(a ** 2 + b ** 2)
    h = np.degrees(np.arctan2(b, a))
    h = np.where(h < 0, h + 360.0, h)

    lch_img = np.stack([L, C, h], axis=-1)
    return lch_img


def _gamma_to_linear(c: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    return c ** gamma

def rgb_to_xyz(img: np.ndarray) -> np.ndarray:

    img = img.astype(np.float64)
    img = img / 255.0

    # Matriz CIE RGB -> XYZ. Sus filas son las ecuaciones de X, Y, Z
    # en función de (R, G, B), por lo que hay que aplicarla como
    # matrix @ [R, G, B] por píxel (de ahí el matrix.T al multiplicar
    # por la imagen vista como vectores fila).

    #matriz 1
    matrix = np.array([
        [0.490, 0.310, 0.200],
        [0.177, 0.813, 0.011],
        [.000, .010, 0.990],
    ])


    rgb_linear = _gamma_to_linear(img)

    xyz = rgb_linear @ matrix.T

    return xyz


# esta son las inversas de las funciones anteriorres
def _linear_to_gamma(c: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    # se recorta en 0 para evitar que valores negativos 
    c = np.clip(c, 0.0, None)
    return c ** (1.0 / gamma)

def xyz_to_rgb(xyz: np.ndarray) -> np.ndarray:
    # matriz original 
    matrix = np.array([
        [0.490, 0.310, 0.200],
        [0.177, 0.813, 0.011],
        [.000, .010, 0.990],
    ])
    
    
    inv_matrix = np.linalg.inv(matrix)
    rgb_linear = xyz @ inv_matrix.T
    
    
    img_norm = _linear_to_gamma(rgb_linear)
    
    # des-normalizamos al rango [0, 255]
    img = img_norm * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)

def lab_to_rgb(img: np.ndarray) -> np.ndarray:
    # parametros base
    _X_n = 1.0
    _Y_n = 1.0
    _Z_n = 1.0
    
    delta = 6.0 / 29.0

    L = img[:, :, 0]
    a = img[:, :, 1]
    b = img[:, :, 2]

    # invertimos para obtener fy, fx y fz
    fy = (L + 16.0) / 116.0
    fx = (a / 500.0) + fy
    fz = fy - (b / 200.0)

    # definimos la función inversa f^-1(t)
    def f_inv(t):
        return np.where(
            t > delta, 
            t ** 3, 
            3.0 * (delta ** 2) * (t - 4.0 / 29.0)
        )

    #obtenemos las coordenadas normalizadas
    xr = f_inv(fx)
    yr = f_inv(fy)
    zr = f_inv(fz)

    # des-normalizacion
    X = xr * _X_n
    Y = yr * _Y_n
    Z = zr * _Z_n

    xyz = np.stack([X, Y, Z], axis=-1)
    
    
    rgb_img = xyz_to_rgb(xyz)
    return rgb_img

def lch_to_rgb(lch_img: np.ndarray, C_prima) -> np.ndarray:
    # se extraen los canales L (luminosidad), C (croma) y h (hue/Tono)
    L = lch_img[:, :, 0]
    C = C_prima
    h = lch_img[:, :, 2]

    # Convertir el ángulo h de grados a radianes 
    h_rad = np.radians(h)

    # Transformación de coordenadas polares (C, h) a cartesianas (a, b)
    a = C * np.cos(h_rad)
    b = C * np.sin(h_rad)

    # Reconstruimos la imagen en el espacio de color LAB
    lab_img = np.stack([L, a, b], axis=-1)

    # finalmente, convertimos de LAB a RGB
    rgb_img = lab_to_rgb(lab_img)

    return rgb_img


