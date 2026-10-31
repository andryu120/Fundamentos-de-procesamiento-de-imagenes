import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt


from procesamiento import (image, notch_filter, mostrar_imagen, espectro_log, pick_points)

def mostrar_valores(imagen,puntos, radio, mode, pad):
    "Recibe una lista [(col_k, row_k)], el modo y la cantidad de zeros para el padding, ademas del tipo de filtro. Retorna la imagen modificada, el espectro original, el espectro modificado"

    rows, cols = imagen.shape

    #aplicar o no padding

    if mode == 'zeros':
        img_padded = np.pad(imagen, pad, mode='constant', constant_values=0)
    elif mode in ('reflect', 'edge'):
        img_padded = np.pad(imagen, pad, mode=mode)
    else:
        pad = 0
        img_padded = imagen.copy()
        

    
    # Transformada de Fourier 
    f_transform = np.fft.fft2(img_padded)
    f_shift = np.fft.fftshift(f_transform)

    # Coordenadas y matriz de distancia respecto al centro
    padded_rows, padded_cols = img_padded.shape
    crow, ccol = padded_rows // 2, padded_cols // 2

    mask_notch = np.ones((padded_rows, padded_cols))

    U, V = np.indices((padded_rows, padded_cols))

    
    fx, fy = puntos[0]
    col_k = ccol + fx * padded_cols
    row_k = crow + fy * padded_rows
    col_k_ , row_k_ = (2*ccol - col_k, 2*crow - row_k)

    Dk = ((U - row_k)**2 + (V - col_k)**2)
    D_menos_k = ((U - row_k_)**2 + (V - col_k_)**2)

    mask_notch[np.sqrt(Dk) <= radio] = 0
    mask_notch[np.sqrt(D_menos_k) <= radio] = 0

    
    idx_row, idx_col = int(row_k), int(col_k)
    
    valor_original = f_shift[idx_row, idx_col]
    valor_H = mask_notch[idx_row, idx_col]
    print(f'Frecuencias: {col_k,row_k}')
    print(f"F(u,v): {valor_original}")
    print(f"H(u,v): {valor_H}")
    print(f"Resultado: {valor_original * valor_H}")
            

def show_imgs(img_og, img_filt, fft_og, fft_filt, mascara):

    
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    
    # imagen original
    axs[0, 0].imshow(img_og, cmap='gray', vmin=0, vmax=255)
    axs[0, 0].set_title('Imagen Ruidosa')
    axs[0, 0].axis('off')
    
    # imagen filt
    axs[0, 1].imshow(img_filt, cmap='gray', vmin=0, vmax=255)
    axs[0, 1].set_title('Imagen Filtrada')
    axs[0, 1].axis('off')
    
    # Esta es la imagen diferencia para ver el error entre imagenes
    diferencia = img_og.astype(float) - img_filt.astype(float)
    
    
    im_diff = axs[0, 2].imshow(diferencia, cmap='gray', vmin=-50, vmax=50) 
    axs[0, 2].set_title('Imagen Diferencia')
    axs[0, 2].axis('off')
    fig.colorbar(im_diff, ax=axs[0, 2], fraction=0.046, pad=0.04)

    # espectro og
    axs[1, 0].imshow(np.log1p(np.abs(fft_og)), cmap='gray')
    axs[1, 0].set_title('Espectro Original')
    axs[1, 0].axis('off')
    
    # espectro filtrado
    axs[1, 1].imshow(np.log1p(np.abs(fft_filt)), cmap='gray')
    axs[1, 1].set_title('Espectro Filtrado')
    axs[1, 1].axis('off')
    
    # mascara
    axs[1, 2].imshow(mascara)
    axs[1, 2].set_title('Mascara')
    axs[1, 2].axis('off')
    plt.tight_layout()
    plt.show()



def comparar_bordes(diccionario_imagenes, size=80):
    fig, axs = plt.subplots(1, len(diccionario_imagenes), figsize=(18, 5))
    
    for ax, (titulo, img) in zip(axs, diccionario_imagenes.items()):
        
        recorte = img[0:size, 0:size]
        
        ax.imshow(recorte, cmap='gray', vmin=0, vmax=255)
        ax.set_title(titulo)
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    imagen = image("cameraman.jpg")
    imagen = image("cameraman.jpg")
    r, c = imagen.shape
    y, x = np.mgrid[0:r, 0:c]

    imagen_ruidosa = imagen + 5*np.sin(2*np.pi*(0.10*x + 0.06*y)) + 5*np.sin(2*np.pi*(0.25*x - 0.18*y))
    puntos = puntos = pick_points(imagen_ruidosa, 256, n=2)
    

    db = espectro_log(np.fft.fftshift(np.fft.fft2(imagen_ruidosa)))
    mostrar_imagen(db, "espectro original")
    

    #2
    for radio in [1]:

        print("Filtro ideal con radio",radio)

        img_modificada, fft_og, fft_modificada, mascara = notch_filter(imagen_ruidosa, puntos, radio, 'edge', 256, "ideal", 4)
        show_imgs(imagen_ruidosa, img_modificada, fft_og, fft_modificada, mascara)

        print("Filtro Gaussiano con radio",radio)
        img_modificada, fft_og, fft_modificada, mascara = notch_filter(imagen_ruidosa, puntos, radio, 'reflect', 256, "gauss", 4)
        show_imgs(imagen_ruidosa, img_modificada, fft_og, fft_modificada, mascara)


    
    # 3
    #Sin padding
    radio = 5.5
    print("Analisis padding")

    comparacion_padding = {}

    print("Filtro sin padding")
    img_modificada, fft_og, fft_modificada, mascara = notch_filter(imagen_ruidosa, puntos, radio, None, 0, "gauss", 4)
    show_imgs(imagen_ruidosa, img_modificada, fft_og, fft_modificada, mascara)
    comparacion_padding["Sin Padding"] = img_modificada

    # print("Filtro con padding 256")
    img_modificada, fft_og, fft_modificada, mascara = notch_filter(imagen_ruidosa, puntos, radio, 'edge', 256, "gauss", 4)
    show_imgs(imagen_ruidosa, img_modificada, fft_og, fft_modificada, mascara)
    comparacion_padding["Padding edge 256"] = img_modificada

    print("Filtro con padding 128")
    img_modificada, fft_og, fft_modificada, mascara = notch_filter(imagen_ruidosa, puntos, radio, 'edge', 128, "gauss", 4)
    show_imgs(imagen_ruidosa, img_modificada, fft_og, fft_modificada, mascara)
    comparacion_padding["Padding Edge 128"] = img_modificada

    print("Filtro con padding (constant)")
    img_modificada, fft_og, fft_modificada, mascara = notch_filter(imagen_ruidosa, puntos, radio, 'zeros', 256, "gauss", 4)
    show_imgs(imagen_ruidosa, img_modificada, fft_og, fft_modificada, mascara)
    comparacion_padding["Padding constant"] = img_modificada

    print("Filtro con padding (reflect) 256")
    img_modificada, fft_og, fft_modificada, mascara = notch_filter(imagen_ruidosa, puntos, radio, 'reflect', 256, "gauss", 4)
    show_imgs(imagen_ruidosa, img_modificada, fft_og, fft_modificada, mascara)
    comparacion_padding["Padding reflect 256"] = img_modificada

    comparar_bordes(comparacion_padding)

    diferencia_padding = np.abs(comparacion_padding["Sin Padding"] - comparacion_padding["Padding edge 256"])

    plt.figure(figsize=(8,8))
    # Amplificar el ruido
    plt.imshow(diferencia_padding * 10, cmap='gray')
    plt.axis('off')
    plt.show()

    # Para pregunta guia
    mostrar_valores(imagen_ruidosa,puntos, radio, 'edge', 256)



    

    

    
