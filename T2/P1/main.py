import numpy as np
import cv2
from creacion_imagen import (imagen_ruidosa,image, mascara_circulo,mascara_exterior,mascara_rectangulo)
from procesamiento import (filtro_gaussiano, mostrar_imagen, rmse)
import matplotlib.pyplot as plt





def graficar_error(imagen_original, imagen_ruidosa):

    lista = [round(i, 1) for i in np.linspace(0.0, 4.0, 21)]
        #valores de sigma
    valores = sorted(list(set(lista)))
    
        
    
    lista_error_global = []
    lista_error_R1 = []
    lista_error_R2 = []
    lista_error_R3 = []

    for valor in valores:
        if valor == 0.0:
            imagen_filtrada = imagen_ruidosa
        else:
            imagen_filtrada = filtro_gaussiano(imagen_ruidosa, valor)

        error_global = rmse(imagen_original, imagen_filtrada)
        lista_error_global.append(error_global)
        
        error_R1 = rmse(imagen_original[mascara_exterior], imagen_filtrada[mascara_exterior])
        lista_error_R1.append(error_R1)
    
        error_R2 = rmse(imagen_original[mascara_rectangulo], imagen_filtrada[mascara_rectangulo])
        lista_error_R2.append(error_R2)

        error_R3 = rmse(imagen_original[mascara_circulo], imagen_filtrada[mascara_circulo])
        lista_error_R3.append(error_R3)
    
        minimo_global = np.argmin(lista_error_global)
        minimo_R1 = np.argmin(lista_error_R1)
        minimo_R2 = np.argmin(lista_error_R2)
        minimo_R3 = np.argmin(lista_error_R3)
    

    #minimos (en el eje x)

    minimo_global = np.argmin(lista_error_global)
    minimo_R1 = np.argmin(lista_error_R1)
    minimo_R2 = np.argmin(lista_error_R2)
    minimo_R3 = np.argmin(lista_error_R3)


    #graficar
    
    #graficar valores de sigma
    plt.figure(figsize=(10, 6))

    label_global = f'Global (minimo RMSE: {round(lista_error_global[minimo_global], 2)})'
    label_r1 = f'Fondo (minimo RMSE: {round(lista_error_R1[minimo_R1], 2)})'
    label_r2 = f'Cuadrado (minimo RMSE: {round(lista_error_R2[minimo_R2], 2)})'
    label_r3 = f'Círculo (minimo RMSE: {round(lista_error_R3[minimo_R3], 2)})'

    # 2. Asignar las etiquetas al parámetro label de plt.plot()
    plt.plot(valores, lista_error_global, label=label_global, color='black', linestyle='--')
    plt.plot(valores, lista_error_R1, label=label_r1, color='blue')
    plt.plot(valores, lista_error_R2, label=label_r2, color='green')
    plt.plot(valores, lista_error_R3, label=label_r3, color='red')

    #marcar valores minimos
    plt.scatter(valores[minimo_global], lista_error_global[minimo_global], color='black', s=100, zorder=5)
    plt.scatter(valores[minimo_R1], lista_error_R1[minimo_R1], color='blue', s=100, zorder=5)
    plt.scatter(valores[minimo_R2], lista_error_R2[minimo_R2], color='green', s=100, zorder=5)
    plt.scatter(valores[minimo_R3], lista_error_R3[minimo_R3], color='red', s=100, zorder=5)




    plt.title('RMSE vs Sigma')
    plt.xlabel('Sigma')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()

    return (lista_error_global[minimo_global], lista_error_R1[minimo_R1], lista_error_R2[minimo_R2], lista_error_R3[minimo_R3])




    
if __name__ == "__main__":


    minimos = graficar_error(image, imagen_ruidosa)

    print(minimos)


    

    
    

       

