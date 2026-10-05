import numpy as np
import cv2
import scipy.ndimage as ndimage

def difusion_anistropica(imagen, lambdda, num_iter, epsilon, K, modo):
    imagen = imagen.copy().astype(float)
    # verificar estabilidad
    # if modo == "variacion total":
    #     assert lambdda <= epsilon/4
    # elif modo in ["laplaciano", "laplaciano racional"]:
    #     assert lambdda <= 1/4


    for iter in range(num_iter):

        # se calcula la derivada direccional
        padding = np.pad(imagen, 1 , mode = 'edge' ) # esto es para evitar problemas con los bordes

        # 4 direcciones

        N = padding[:-2, 1:-1] - imagen
        S = padding[2:, 1: -1] - imagen
        E = padding[1:-1 , 2:] - imagen
        O = padding[1: -1, : -2] - imagen

        if modo == "variacion total":
    
            cN, cS, cE , cO = variacion_total(imagen, epsilon)

        elif modo == "laplaciano":
            


            cN, cS, cE , cO = laplaciano(imagen, K)

        elif modo == "laplaciano racional":

            cN, cS, cE , cO = gradiente_rac(imagen, K)

        # por ultimo se actualiza la imagen

        imagen = imagen + lambdda * (cN * N + cS * S + cE * E + cO * O)

        # se guarda el mapa de coef

        if iter == (num_iter - 1):
            mapa_c = (cN + cS + cE + cO) / 4.0


    return imagen, mapa_c



def variacion_total(img, epsilon):

    # calculo gradiente
    dy, dx = np.gradient(img)
    magnitud_gradiente = dx**2 + dy**2

    c_val = 1.0 / np.sqrt(magnitud_gradiente + epsilon**2)
   
    cN =  cS = cE = cO = c_val

    return cN, cS, cE , cO

def laplaciano(img, K):
    img = img.copy()
    img_suavizada = ndimage.gaussian_filter(img, sigma=1.2)
            
    padded_suav = np.pad(img_suavizada, 1, mode='edge')
    N_suav = padded_suav[:-2, 1:-1] - img_suavizada
    S_suav = padded_suav[2:, 1:-1] - img_suavizada
    E_suav = padded_suav[1:-1, 2:] - img_suavizada
    O_suav = padded_suav[1:-1, :-2] - img_suavizada
    
    # Calculamos el Laplaciano discreto
    laplaciano = N_suav + S_suav + E_suav + O_suav
            
    
    magnitud_laplace = np.abs(laplaciano)
    
    # Coeficiente de difusion, se modifica la exponencial
    c_val = np.exp(-(magnitud_laplace / K)**2)
    cN = cS = cE = cW = c_val # c toma el mismo valor en todas las direcciones del pixel central

    return cN , cS , cE , cW

def gradiente_rac(img,K):
    # aplicamos un suavizado para evitar artefactos
    img = img.copy()
    img_suavizada = ndimage.gaussian_filter(img, sigma=1.0)
            
    dy , dx = np.gradient(img_suavizada)

   
    
    
    magnitud_gradiente =  np.sqrt(dx**2 + dy**2)
    
    # funcion modificada
    potencia = 2
    c_val = 1.0 / (1.0 + (magnitud_gradiente / K)**potencia)
    
    cN = cS = cE = cO = c_val

    return cN , cS , cE , cO



def mostrar_imagen(img: np.ndarray, titulo: str):
    import matplotlib.pyplot as plt
    # # Si queremos mostrala
    
    plt.figure(figsize=(15,8))
    plt.imshow(img, cmap='gray')
    plt.title(f'{titulo}')
    plt.show()


def rmse(imagen_referencia: np.ndarray, imagen_estimada: np.ndarray) -> float:
    """
    Calcula el RMSE  entre una imagen de referencia y una imagen estimada.

    Parámetros:
        imagen_referencia : np.ndarray, imagen limpia (ground truth)
        imagen_estimada    : np.ndarray, imagen filtrada a evaluar
    """
    referencia = imagen_referencia.astype(np.float64)
    estimada = imagen_estimada.astype(np.float64)
    return float(np.sqrt(np.mean((referencia - estimada) ** 2)))

