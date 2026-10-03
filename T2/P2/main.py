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


if __name__ == "__main__":

    #1
    # epsilons = [1e-1, 1e-2, 1e-3, 1e-4]
    # mostrar_imagen(imagen_ruidosa, "imagen ruidosa")
    # for ep in epsilons: 
    #     lam = ep / 5.0  # para que este dentro del rango
    #     num_iters = int(1.0 / lam) # numero a probar que depende de lambda y epsilon

    #     imagen_mod_tv, mapa_c = difusion_anistropica(imagen_ruidosa, lam, num_iters, ep , 0, "variacion total")  # nose que poner en los parametros
    #     mostrar_imagen(imagen_mod_tv, f"imagen con parametros lam = {lam}, iteraciones = {num_iters}, epsilon = {ep}")
    #     graficar_mapa(mapa_c ,f"Mapa c parametros lam = {lam}, iteraciones = {num_iters}, epsilon = {ep}")



    # 2 se podria hacer un diccionario con los valores a probar? maybe, estas estudiarlas y registrar los casos solicitados

    # casos_fallo = {
    #     "Lento": {"ep": 0.1, "lam": 0.001, "num_iter": 5}, 
    #     "Sobre-suavizado": {"ep": 0.5, "lam": 0.1, "num_iter": 200},
    #     "Inestabilidad Numerica": {"ep": 1e-3, "lam": 0.5, "num_iter": 10}
    # }

    # for titulo, parametros in casos_fallo.items():
    #     imagen_mod_tv, mapa_c = difusion_anistropica(imagen_ruidosa, parametros["lam"], parametros["num_iter"], parametros["ep"] , 0, "variacion total")
    #     print(f'{titulo}')
    #     mostrar_imagen(imagen_mod_tv, f"imagen con parametros lam = {parametros["lam"]}, iteraciones = {parametros["num_iter"]}, epsilon = {parametros["ep"]}")


    # 3 probar los coeficientes propuestos junto a sus parametros, puede ser un for probando parametros

    for k in [0.01, 0.05, 0.1, 0.5]:
        # con epsilon 0 y lambda 0.2, 30 repeticiones
        print("laplaciano con exp")
        img_mod_lap, mapa_c_lap = difusion_anistropica(imagen_ruidosa, 0.2, 30, 0, k, "laplaciano")
        mostrar_imagen(img_mod_lap,f"imagen con parametros lam = {0.2}, iteraciones = {30}, epsilon = {0}, k = {k}" )
        graficar_mapa(mapa_c_lap, f"Mapa c parametros lam = {0.2}, iteraciones = {30}, epsilon = {0}, k = {k}")
        print("laplaciano con frac")
        img_mod_lap_frac, mapa_c_lap_frac = difusion_anistropica(imagen_ruidosa, 0.2, 30, 0, k, "laplaciano racional")
        mostrar_imagen(img_mod_lap,f"imagen con parametros lam = {0.2}, iteraciones = {30}, epsilon = {0}, k = {k}" )
        graficar_mapa(mapa_c_lap_frac, f"Mapa c con parametros lam = {0.2}, iteraciones = {30}, epsilon = {0}, k = {k}")

        

    # 4 tengo que analizar las dos propuestas ya que ambas ocupan el laplaciano y falta implementar el suavizado en la de fraccion


    # 5 comparar los tres metodos con los mismos parametros, tambien incluir RMSE respecto a la imagen original
    # Parametros a utilizar
    tv_ep = 0.01
    tv_lam = 0.002
    tv_iter = 150
    
    
    lap_k = 0.05
    lap_lam = 0.2
    lap_iter = 40

    image = image / 255.0
    print("Filtrando con Variación Total...")
    img_tv, _ = difusion_anistropica(imagen_ruidosa, tv_lam, tv_iter, tv_ep, 0, "variacion total")
    
    print("Filtrando con Laplaciano Exponencial...")
    img_exp, _ = difusion_anistropica(imagen_ruidosa, lap_lam, lap_iter, 0, lap_k, "laplaciano")
    
    print("Filtrando con Laplaciano Racional...")
    img_rac, _ = difusion_anistropica(imagen_ruidosa, lap_lam, lap_iter, 0, lap_k, "laplaciano racional")

    # calculo rmse
    
    err_ruido = rmse(image, imagen_ruidosa)
    err_tv = rmse(image, img_tv)
    err_exp = rmse(image, img_exp)
    err_rac = rmse(image, img_rac)

    print(f"RMSE Imagen Ruidosa: {err_ruido:.4f}")
    print(f"RMSE Variacion Total: {err_tv:.4f}")
    print(f"RMSE Lap. Exponencial: {err_exp:.4f}")
    print(f"RMSE Lap. Racional: {err_rac:.4f}")

    # esto es para el crop
    y1, y2 = 100, 150
    x1, x2 = 45, 85 

    imagenes = [image, imagen_ruidosa, img_tv, img_exp, img_rac]
    titulos = [
        "Original", 
        f"Ruidosa\nRMSE: {err_ruido:.4f}", 
        f"TV (eps={tv_ep})\nRMSE: {err_tv:.4f}", 
        f"Lap. Exp (K={lap_k})\nRMSE: {err_exp:.4f}", 
        f"Lap. Rac (K={lap_k})\nRMSE: {err_rac:.4f}"
    ]

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))

    for i in range(5):
        
        axes[0, i].imshow(imagenes[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(titulos[i], fontsize=10)
        axes[0, i].axis('off')

        # este es el recorte
        crop = imagenes[i][y1:y2, x1:x2]
        axes[1, i].imshow(crop, cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title("Recorte del borde", fontsize=9)
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()



    

    # 6 esta es para el informe, importa la parte matematica 
    