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

def transformacion_hsv(img: np.ndarray, m_base: np.ndarray):
    # para transformar la matriz original, se va a multiplicar por esta nueva matriz de parametros m
    # Saturacion
    
    S = img[:, :, 1]

    S_prima = S * m_base
    return S_prima

def transformacion_lcab(img: np.ndarray, m_base: np.ndarray):
    
    
    a = img[:, :, 1]
    b = img[:, :, 2]

    C = np.sqrt(a ** 2 + b ** 2)
    C_prima = C * m_base
    return C_prima
    


#actualizacion de la imagen con la nueva saturacion


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

def _gamma_to_linear(c: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    return c ** gamma

def rgb_to_xyz(img: np.ndarray) -> np.ndarray:

    img = img.astype(np.float64)
    img = img / 255.0

    # Matriz CIE RGB -> XYZ. Sus filas son las ecuaciones de X, Y, Z
    # en función de (R, G, B), por lo que hay que aplicarla como
    # matrix @ [R, G, B] por píxel (de ahí el matrix.T al multiplicar
    # por la imagen vista como vectores fila).
    matrix = np.array([
        [0.490, 0.310, 0.200],
        [0.177, 0.813, 0.011],
        [.000, .010, 0.990],
    ])

    rgb_linear = _gamma_to_linear(img)

    xyz = rgb_linear @ matrix.T

    return xyz

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



def _linear_to_gamma(c: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    # Se recorta en 0 para evitar que valores negativos (por errores de redondeo 
    # o colores fuera de gama) generen "NaN" al elevar a una potencia fraccionaria.
    c = np.clip(c, 0.0, None)
    return c ** (1.0 / gamma)

def xyz_to_rgb(xyz: np.ndarray) -> np.ndarray:
    # Matriz original definida en tu rgb_to_xyz
    matrix = np.array([
        [0.490, 0.310, 0.200],
        [0.177, 0.813, 0.011],
        [.000, .010, 0.990],
    ])
    
    # Calculamos la matriz inversa para pasar de XYZ a RGB
    inv_matrix = np.linalg.inv(matrix)

    # Multiplicación matricial inversa por píxel
    rgb_linear = xyz @ inv_matrix.T
    
    # Aplicamos la inversa de la corrección gamma
    img_norm = _linear_to_gamma(rgb_linear)
    
    # Des-normalizamos al rango [0, 255]
    img = img_norm * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)

def lab_to_rgb(img: np.ndarray) -> np.ndarray:
    # Los mismos parámetros base
    _X_n = 1.0
    _Y_n = 1.0
    _Z_n = 1.0
    
    delta = 6.0 / 29.0

    L = img[:, :, 0]
    a = img[:, :, 1]
    b = img[:, :, 2]

    # Invertimos las fórmulas para obtener fy, fx y fz
    fy = (L + 16.0) / 116.0
    fx = (a / 500.0) + fy
    fz = fy - (b / 200.0)

    # Definimos la función inversa f^-1(t)
    def f_inv(t):
        return np.where(
            t > delta, 
            t ** 3, 
            3.0 * (delta ** 2) * (t - 4.0 / 29.0)
        )

    # Obtenemos las coordenadas normalizadas
    xr = f_inv(fx)
    yr = f_inv(fy)
    zr = f_inv(fz)

    # Des-normalizamos por el iluminante
    X = xr * _X_n
    Y = yr * _Y_n
    Z = zr * _Z_n

    xyz = np.stack([X, Y, Z], axis=-1)
    
    # Encadenamos con la función XYZ a RGB
    rgb_img = xyz_to_rgb(xyz)
    return rgb_img

def lch_to_rgb(lch_img: np.ndarray, C_prima) -> np.ndarray:
    # Se extraen los canales L (Luminosidad), C (Croma) y h (Hue/Tono)
    L = lch_img[:, :, 0]
    C = C_prima
    h = lch_img[:, :, 2]

    # Convertir el ángulo h de grados a radianes para las funciones trigonométricas
    h_rad = np.radians(h)

    # Transformación de coordenadas polares (C, h) a cartesianas (a, b)
    a = C * np.cos(h_rad)
    b = C * np.sin(h_rad)

    # Reconstruimos la imagen en el espacio de color LAB
    lab_img = np.stack([L, a, b], axis=-1)

    # Finalmente, convertimos de LAB a RGB
    # Nota: Requiere que tengas definida la función lab_to_rgb()
    rgb_img = lab_to_rgb(lab_img)

    return rgb_img


def mostrar_imagen(img: np.ndarray):
    # # Si queremos mostrala
    plt.figure(figsize= (15,8))
    plt.imshow(img)
    plt.show()




def color_saturation(name: str, path: str, p: list, modo: str):
    if modo == ("HSV" or "hsv"):
        
        imagen_original, imagen = image(name, path)
        imagen_hsv = rgb_to_hsv(imagen)
        p = correcion_m(p)
        M = interpolar(imagen_hsv, p)
        S_prima = transformacion_hsv(imagen_hsv,M)
        imagen_rgb = hsv_to_rgb(imagen_hsv, S_prima)
        return imagen_rgb

    elif modo == ("CIE" or "cie" or " CIE L*c*h*"):
        
        imagen_original, imagen = image(name, path)
        imagen_lch = rgb_to_lch(imagen)
        p = correcion_m(p)
        M = interpolar(imagen_lch, p)
        C_prima = transformacion_lcab(imagen_lch,M)
        imagen_rgb = lch_to_rgb(imagen_lch, C_prima)
        return imagen_rgb
    else:
        raise ValueError("Modo incorrecto, ingrese de nuevo el modo")
    