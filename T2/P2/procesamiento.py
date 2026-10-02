import numpy as np
import cv2
import scipy


def difusion_anistropica(imagen, lambdda, num_iter, epsilon, K, modo):
    imagen = imagen.copy().astype(float)

    for iter in range(num_iter):

        # se calcula la derivada direccional
        padding = np.pad(imagen, 1 , mode = 'edge' ) # esto es para evitar problemas con los bordes

        # 4 direcciones

        N = padding[:-2, 1:-1] - imagen
        S = padding[2:, 1: -1] - imagen
        E = padding[1:-1 , 2:] - imagen
        O = padding[1: -1, : -2] - imagen

        if modo == "variacion total":

            # antes de continuar verificamos la estabilidad

            if lambdda > epsilon/4:
                lambdda = epsilon/4 - 0.01 # se reduce para que pueda correr
                print(f"Se redujo lambda a {lambdda}")

            cN, cS, cE , cO = variacion_total(N, S, E, O, epsilon)

        elif modo == "laplaciano":
            # verificar estabilidad
            if lambdda > 1/4:
                lambdda = 1/4 - 0.01
                print(f"Se redujo lambda a {lambdda}")

            cN, cS, cE , cO = laplaciano(N, S, E, O, K)

        elif modo == "laplaciano racional":
            if lambdda > 1/4:
                lambdda = 1/4 - 0.01
                print(f"Se redujo lambda a {lambdda}")

            cN, cS, cE , cO = racional_laplaciano(N, S, E, O, K)

        # por ultimo se actualiza la imagen

        imagen = imagen + lambdda * (cN * N + cS * S + cE * E + cO * O)

    return imagen



def variacion_total(N, S, E, O, epsilon):

    cN = 1.0 / np.sqrt(N**2 + epsilon**2)
    cS = 1.0 / np.sqrt(S**2 + epsilon**2)
    cE = 1.0 / np.sqrt(E**2 + epsilon**2)
    cO = 1.0 / np.sqrt(O**2 + epsilon**2)

    return cN, cS, cE , cO

def laplaciano(N, S, E, O, K):
    laplaciano = N + S + E + O
            
    
    magnitud_laplace = np.abs(laplaciano)
    
    # Coeficiente de difusion, se modifica la exponencial
    c_val = np.exp(-(magnitud_laplace / K)**2)
    cN = cS = cE = cW = c_val # c toma el mismo valor en todas las direcciones del pixel central

    return cN , cS , cE , cW

def racional_laplaciano(N, S, E, O, K):
    laplaciano = N + S + E + O
    magnitud_laplace = np.abs(laplaciano)

    potencia = 4 # se ocupa una potencia de 4 en vez de 2
    c_val = 1.0 / (1.0 + (magnitud_laplace / K)**potencia)

    cN = cS = cE = cO = c_val

    return cN , cS , cE , cO



def mostrar_imagen(img: np.ndarray, titulo: str):
    import matplotlib.pyplot as plt
    # # Si queremos mostrala
    
    plt.figure(figsize=(15,8))
    plt.imshow(img, cmap='gray')
    plt.title(f'{titulo}')
    plt.show()

