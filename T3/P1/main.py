import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt


from procesamiento import (image, notch_filter, mostrar_imagen, espectro_log ,pick_peaks)


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








if __name__ == "__main__":
    imagen = image("cameraman.jpg")
    r, c = imagen.shape
    y, x = np.mgrid[0:r, 0:c]
    imagen_ruidosa = imagen + 30*np.sin(2*np.pi*(0.10*x + 0.06*y)) + 20*np.sin(2*np.pi*(0.25*x - 0.18*y))

    db = espectro_log(np.fft.fftshift(np.fft.fft2(imagen_ruidosa)))

    mostrar_imagen(db, "espectro original")
    puntos = pick_peaks(imagen_ruidosa, 256, n=2)

    #2
    for radio in [1, 6, 10, 15]:

        print("Filtro ideal con radio",radio)

        img_modificada, fft_og, fft_modificada, mascara = notch_filter(imagen_ruidosa, radio, 'edge', 256, "ideal", 4)
        show_imgs(imagen_ruidosa, img_modificada, fft_og, fft_modificada, mascara)

        print("Filtro Gaussiano con radio",radio)
        img_modificada, fft_og, fft_modificada, mascara = notch_filter(imagen_ruidosa, radio, 'edge', 256, "gauss", 4)
        show_imgs(imagen_ruidosa, img_modificada, fft_og, fft_modificada, mascara)


    
    # 3
    #Sin padding
    radio = 6
    print("Analisis padding")

    print("Filtro sin padding")
    img_modificada, fft_og, fft_modificada, mascara = notch_filter(imagen_ruidosa, radio, None, 0, "gauss", 4)
    show_imgs(imagen_ruidosa, img_modificada, fft_og, fft_modificada, mascara)

    print("Filtro con padding 256")
    img_modificada, fft_og, fft_modificada, mascara = notch_filter(imagen_ruidosa, radio, 'edge', 256, "gauss", 4)
    show_imgs(imagen_ruidosa, img_modificada, fft_og, fft_modificada, mascara)

    print("Filtro con padding 128")
    img_modificada, fft_og, fft_modificada, mascara = notch_filter(imagen_ruidosa, radio, 'edge', 128, "gauss", 4)
    show_imgs(imagen_ruidosa, img_modificada, fft_og, fft_modificada, mascara)

    

    

    
