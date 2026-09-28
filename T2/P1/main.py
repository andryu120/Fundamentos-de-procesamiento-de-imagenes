import numpy as np
import cv2
from creacion_imagen import (imagen_ruidosa,image, mascara_circulo,mascara_exterior,mascara_rectangulo)
from procesamiento import (filtro_gaussiano, mostrar_imagen, rmse , filtro_gaussiano_adap)
import matplotlib.pyplot as plt





def graficar_error(imagen_original, imagen_ruidosa, graficar: str):
    imagen_original = imagen_original.copy()
    imagen_ruidosa = imagen_ruidosa.copy()

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
        
        error_exterior = rmse(imagen_original[mascara_exterior], imagen_filtrada[mascara_exterior])
        lista_error_R1.append(error_exterior)
    
        error_recangulo = rmse(imagen_original[mascara_rectangulo], imagen_filtrada[mascara_rectangulo])
        lista_error_R2.append(error_recangulo)

        error_circulo = rmse(imagen_original[mascara_circulo], imagen_filtrada[mascara_circulo])
        lista_error_R3.append(error_circulo)
    
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
    if graficar in ["Si", "si"]:
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



def comparacion(imagen,imagen_adaptativa, valores_minimos):
    imagen_adaptativa = imagen_adaptativa.copy()
    rmse_adap_global = rmse(imagen, imagen_adaptativa)
    rmse_adap_fondo = rmse(imagen[mascara_exterior], imagen_adaptativa[mascara_exterior])
    rmse_adap_cuad = rmse(imagen[mascara_rectangulo], imagen_adaptativa[mascara_rectangulo])
    rmse_adap_circ = rmse(imagen[mascara_circulo], imagen_adaptativa[mascara_circulo])
    print("Comparacion \n")

    print(f"{'Región':<12} | {'Mejor Global':<15} | {'Adaptativo':<15}  \n")

    print(f"{'Fondo':<12} | {valores_minimos['Minimo Exterior']:<15.2f} | {rmse_adap_fondo:<15.2f}")
    print(f"{'Cuadrado':<12} | {valores_minimos['Minimo Rectangulo']:<15.2f} | {rmse_adap_cuad:<15.2f}")
    print(f"{'Círculo':<12} | {valores_minimos['Minimo Circulo']:<15.2f} | {rmse_adap_circ:<15.2f}")
    print("-" * 48)
    print(f"{'Global':<12} | {valores_minimos['Minimo Global']:<15.2f} | {rmse_adap_global:<15.2f}")
    
    
if __name__ == "__main__":

    #1,2,3
    minimos = graficar_error(image, imagen_ruidosa, 'no')

    valores_minimos = {"Minimo Global" : minimos[0],
                       "Minimo Exterior": minimos[1],
                       "Minimo Rectangulo": minimos[2],
                       "Minimo Circulo": minimos[3]}
    
    #4,5
    imagen_global = filtro_gaussiano(imagen_ruidosa, valores_minimos["Minimo Global"])
    #mostrar_imagen(imagen_global)
    sigma_a_probar = 2
    imagen_adaptativa, mapeo_sigma = filtro_gaussiano_adap(imagen_ruidosa, sigma_a_probar, valores_minimos, "no")
    mostrar_imagen(imagen_adaptativa)
    comparacion(image, imagen_adaptativa, valores_minimos)
    #6
    imagen_adaptativa_interpolada, mapeo_sigma_interpolada = filtro_gaussiano_adap(imagen_ruidosa, sigma_a_probar, valores_minimos, "si")

    
    

    
    
    


    

    
    

       

