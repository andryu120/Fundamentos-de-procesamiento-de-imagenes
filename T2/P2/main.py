import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
from creacion_imagen import (image, imagen_ruidosa)
from procesamiento import (difusion_anistropica, mostrar_imagen , rmse)

def graficar_mapa(map, titulo):
    plt.figure(figsize=(15,8))
    im = plt.imshow(map, cmap='viridis')
    plt.colorbar(im, label='magnitud del coeficiente c', shrink=0.8)
    plt.title(f'{titulo}')
    plt.axis('off') # Quita los ejes (coordenadas x,y) para un reporte más limpio
    plt.tight_layout()
    plt.show()


def seleccionar_pixel_var_total(img):
    img = img.copy()

    epsilon_test = 0.001
    lambdda_test = 0.0002


    # Se realiza el mismo procedimiento pero solamente una iteracion
    padding = np.pad(img, 1, mode='edge')
    N = padding[:-2, 1:-1] - img
    S = padding[2:, 1:-1] - img
    E = padding[1:-1, 2:] - img
    O = padding[1:-1, :-2] - img

    # coords
    # pixel homogeneo
    y_h, x_h = 30, 30
    # pixel borde
    y_b, x_b = 128, 64

    # coef variacion total

    dy, dx = np.gradient(img)
    magnitud_grad_sq = dx**2 + dy**2
    c_mapa = 1.0 / np.sqrt(magnitud_grad_sq + epsilon_test**2)
    for tipo, y, x in [("Fondo", y_h, x_h), ("Borde", y_b, x_b)]:
        dif_N, dif_S, dif_E, dif_O = N[y,x], S[y,x], E[y,x], O[y,x]
        c_pixel = c_mapa[y,x]
        actualizacion = lambdda_test * c_pixel * (dif_N + dif_S + dif_E + dif_O)

        print(f"[{tipo} - y:{y}, x:{x}]")
        print(f"Valor inicial del pixel: {img[y,x]:.4f}")
        print(f"Diferencias direccionales N: {dif_N:.4f}, S: {dif_S:.4f}, E: {dif_E:.4f}, O: {dif_O:.4f}")
        print(f"Valor  c: {c_pixel:.6f}")
        print(f"Actualización a sumar: {actualizacion:.6f}")
        print(f"Nuevo valor del pixel: {img[y,x] + actualizacion:.4f}\n")
    

def seleccionar_pixel_laplace(img):
    img = img.copy()
    lambdda_test = 0.2
    K_test = 0.05
    
    # Se realiza el mismo procedimiento pero solamente una iteracion
    padding = np.pad(img, 1, mode='edge')
    N = padding[:-2, 1:-1] - img
    S = padding[2:, 1:-1] - img
    E = padding[1:-1, 2:] - img
    O = padding[1:-1, :-2] - img
    
    # Coeficientes laplace
    img_suav = scipy.ndimage.gaussian_filter(img, sigma=1.2)
    pad_suav = np.pad(img_suav, 1, mode='edge')
    laplaciano_val = (pad_suav[:-2, 1:-1] - img_suav) + (pad_suav[2:, 1:-1] - img_suav) + (pad_suav[1:-1, 2:] - img_suav) + (pad_suav[1:-1, :-2] - img_suav)
    c_mapa = np.exp(-(np.abs(laplaciano_val) / K_test)**2)
    
    # coords
    # pixel homogeneo
    y_h, x_h = 30, 30
    # pixel borde
    y_b, x_b = 128, 64
    
    for tipo, y, x in [("Fondo", y_h, x_h), ("Borde", y_b, x_b)]:
        dif_N, dif_S, dif_E, dif_O = N[y,x], S[y,x], E[y,x], O[y,x]
        c_pixel = c_mapa[y,x]
        actualizacion = lambdda_test * c_pixel * (dif_N + dif_S + dif_E + dif_O)
        
        print(f"[{tipo} - y:{y}, x:{x}]")
        print(f"Valor inicial del pixel: {img[y,x]:.4f}")
        print(f"Diferencias direccionales N: {dif_N:.4f}, S: {dif_S:.4f}, E: {dif_E:.4f}, O: {dif_O:.4f}")
        print(f"Valor  c: {c_pixel:.6f}")
        print(f"Actualización a sumar: {actualizacion:.6f}")
        print(f"Nuevo valor del pixel: {img[y,x] + actualizacion:.4f}\n")


if __name__ == "__main__":

    # # 1
    # parametros = [(0.1, 0.025,50), (0.05, 0.0125,80), (0.01, 0.0025,100), (0.0001, 2.5e-5 , 3000)]
    
    # mostrar_imagen(imagen_ruidosa, "imagen ruidosa")
    # for par in parametros: 
        

    #     imagen_mod_tv, mapa_c = difusion_anistropica(imagen_ruidosa, par[1], par[2], par[0] , 0, "variacion total")  
    #     mostrar_imagen(imagen_mod_tv, f"imagen con parametros lam = {par[1]}, iteraciones = {par[2]}, epsilon = {par[0]}")
    #     graficar_mapa(mapa_c ,f"Mapa c parametros lam = {par[1]}, iteraciones = {par[2]}, epsilon = {par[0]}")



    # # 2 se podria hacer un diccionario con los valores a probar? 
    # casos_fallo = {
    #     "Lento": {"ep": 0.1, "lam": 0.001, "num_iter": 5}, 
    #     "Sobre-suavizado": {"ep": 0.5, "lam": 0.1, "num_iter": 200},
    #     "Inestabilidad Numerica": {"ep": 1e-3, "lam": 0.5, "num_iter": 10}
    # }

    # for titulo, parametros in casos_fallo.items():
    #     imagen_mod_tv, mapa_c = difusion_anistropica(imagen_ruidosa, parametros["lam"], parametros["num_iter"], parametros["ep"] , 0, "variacion total")
    #     print(f'{titulo}')
    #     mostrar_imagen(imagen_mod_tv, f"imagen con parametros lam = {parametros["lam"]}, iteraciones = {parametros["num_iter"]}, epsilon = {parametros["ep"]}")


    # # 3 probar los coeficientes propuestos junto a sus parametros, puede ser un for probandos parametros

    # for k in [0.01, 0.05, 0.1, 0.5]:
    #     # con epsilon 0 y lambda 0.2, 30 repeticiones
    #     print("laplaciano con exp")
    #     img_mod_lap, mapa_c_lap = difusion_anistropica(imagen_ruidosa, 0.2, 30, 0, k, "laplaciano")
    #     mostrar_imagen(img_mod_lap,f"imagen con parametros lam = {0.2}, iteraciones = {30}, epsilon = {0}, k = {k}" )
    #     graficar_mapa(mapa_c_lap, f"Mapa c parametros lam = {0.2}, iteraciones = {30}, epsilon = {0}, k = {k}")
    #     print("laplaciano con frac")
    #     img_mod_lap_frac, mapa_c_lap_frac = difusion_anistropica(imagen_ruidosa, 0.2, 30, 0, k, "laplaciano racional")
    #     mostrar_imagen(img_mod_lap,f"imagen con parametros lam = {0.2}, iteraciones = {30}, epsilon = {0}, k = {k}" )
    #     graficar_mapa(mapa_c_lap_frac, f"Mapa c con parametros lam = {0.2}, iteraciones = {30}, epsilon = {0}, k = {k}")

        

    # 4 tengo que analizar las dos propuestas ya que ambas ocupan el laplaciano y falta implementar el suavizado en la de fraccion


    # 5 comparar los tres metodos con los mismos parametros, tambien incluir RMSE respecto a la imagen original
    # Parametros a utilizar
    # tv_ep = 0.0001
    # tv_lam = 2.5e-5
    # tv_iter = 3000
    
    
    # lap_k = 0.05
    # lap_lam = 0.2
    # lap_iter = 40

    # image = image / 255.0
    
    # img_tv, _ = difusion_anistropica(imagen_ruidosa, tv_lam, tv_iter, tv_ep, 0, "variacion total")
    
    
    # img_exp, _ = difusion_anistropica(imagen_ruidosa, lap_lam, lap_iter, 0, lap_k, "laplaciano")
    
    
    # img_rac_grad, _ = difusion_anistropica(imagen_ruidosa, lap_lam, lap_iter, 0, lap_k, "laplaciano racional")

    # # calculo rmse
    
    # err_ruido = rmse(image, imagen_ruidosa)
    # err_tv = rmse(image, img_tv)
    # err_exp = rmse(image, img_exp)
    # err_rac_grad = rmse(image, img_rac_grad)

    # # esto es para el crop (recorte)
    # y1, y2 = 100, 150
    # x1, x2 = 45, 85 

    # imagenes = [image, imagen_ruidosa, img_tv, img_exp, img_rac_grad]
    # titulos = [
    #     "Original", 
    #     f"Ruidosa\nRMSE: {err_ruido:.4f}", 
    #     f"TV (eps={tv_ep})\nRMSE: {err_tv:.4f}", 
    #     f"Lap. Exp (K={lap_k})\nRMSE: {err_exp:.4f}", 
    #     f"Lap. Rac Grad (K={lap_k})\nRMSE: {err_rac_grad:.4f}"
    # ]

    # fig, axes = plt.subplots(2, 5, figsize=(18, 7))

    # for i in range(5):
        
    #     axes[0, i].imshow(imagenes[i], cmap='gray', vmin=0, vmax=1)
    #     axes[0, i].set_title(titulos[i], fontsize=10)
    #     axes[0, i].axis('off')

    #     # este es el recorte
    #     crop = imagenes[i][y1:y2, x1:x2]
    #     axes[1, i].imshow(crop, cmap='gray', vmin=0, vmax=1)
    #     axes[1, i].set_title("Recorte del borde", fontsize=9)
    #     axes[1, i].axis('off')

    # plt.tight_layout()
    # plt.show()

    #Preguntas guiadas

    seleccionar_pixel_var_total(imagen_ruidosa)
    seleccionar_pixel_laplace(imagen_ruidosa)



    

    # 6 esta es para el informe, importa la parte matematica 
    