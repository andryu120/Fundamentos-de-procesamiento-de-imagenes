import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from cv2 import equalizeHist

#Importar una imagen
def image(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            ruta = os.path.join(root, name)
    name = ruta
    imagen = cv2.imread(name, cv2.IMREAD_GRAYSCALE) # leer en escalas de grises
    return imagen

# mostrar histograma
def imhist(X):
  (N,M) = X.shape
  n = 256
  h = np.zeros((256,))
  for i in range(N):
    for j in range(M):
      x = X[i,j]
      h[x] = h[x]+1
  plt.figure(figsize=(20,9))
  plt.bar(range(n),h[0:n])
  plt.show()




def malla(img, distancia_y: int, distancia_x: int):
   alto_imagen, ancho_imagen = img.shape[:2]
   # centramos la malla

   centros_x = np.arange(distancia_x // 2, ancho_imagen,distancia_x)
   centros_y = np.arange(distancia_y // 2, alto_imagen,distancia_y)

   #retorna una lista de centros
   return centros_y, centros_x

def extraer_region(img, alto_region: int, ancho_region: int, c_y: float, c_x: float):
   # esta es para un centro especifico
   alto_imagen, ancho_imagen = img.shape[:2]


   y_min = c_y - alto_region // 2
   y_max = c_y + alto_region // 2
   x_min = c_x - ancho_region // 2
   x_max = c_x + ancho_region // 2

   # recortar a los limites reales de la imagen
   y_min = max(0, y_min)
   y_max = min(alto_imagen, y_max)
   x_min = max(0, x_min)
   x_max = min(ancho_imagen, x_max)

   return img[y_min:y_max, x_min:x_max]

   

   
def calculo_cdf(region):
   # calculo para solo una region
   M, n = region.shape[:2]

   n_bins = 256 # cambiar eventualmente
   hist, bin_edges = np.histogram(region,bins = n_bins, range = (0,256), density = False)
   pdf = hist / (M*n)

   cdf = np.cumsum(pdf) # sumatoria
   s_values = np.round((n_bins-1)*cdf).astype(int)

   return s_values

def calculo_cdf_bin(region,n_bins):
   # calculo para solo una region
   M, n = region.shape[:2]

   
   hist, bin_edges = np.histogram(region,bins = n_bins, range = (0,256), density = False)
   pdf = hist / (M*n)

   cdf = np.cumsum(pdf) # sumatoria
   s_values = np.round((n_bins-1)*cdf).astype(int)

   return s_values

def contraste(region, control_limite: float):
   # realiza el mismo calculo cdf pero teniendo en cuenta el contraste
   M, n = region.shape[:2]
   n_bins = 256
   hist, bin_edges = np.histogram(region,bins = n_bins, range = (0,256), density = False)

   # el control de limite va de [1,0.0], son porcentajes
   # cuanto equivale el limite en pixeles
   pixeles_limite = int(control_limite * (M*n))

   exceso_pixeles = 0
   for i in range(len(hist)):
      if hist[i] > pixeles_limite:
         exceso_pixeles += (hist[i]-pixeles_limite)
         hist[i] = pixeles_limite

   hist = hist + (exceso_pixeles / n_bins)  

   pdf = hist / (M*n)
   
   cdf = np.cumsum(pdf) # sumatoria
   s_values = np.round((n_bins-1)*cdf).astype(int)
   
   return s_values
   
         

def mostrar_region_homogenea(img, distancia_y: int, distancia_x: int, alto_region: int, ancho_region: int):
   alto_imagen, ancho_imagen = img.shape[:2]

   centros_y, centros_x = malla(img, distancia_y, distancia_x)
   min_varianza = float('inf')
   for c_y in centros_y:
      for c_x in centros_x:
         region = extraer_region(img, alto_region, ancho_region, c_y, c_x)
         varianza = np.var(region)
         if varianza < min_varianza:
            min_varianza = varianza
            mejor_centro = (c_y, c_x)

   c_y, c_x = mejor_centro
   region = extraer_region(img, alto_region, ancho_region, c_y, c_x)
   M, n = region.shape[:2]
   n_bins = 256
   hist, bin_edges = np.histogram(region,bins = n_bins, range = (0,256), density = False)
   pdf = hist / (M*n)  
   cdf = np.cumsum(pdf) # sumatoria

   #mostrar los histogramas
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    
   ax1.imshow(region, cmap='gray', vmin=0, vmax=255)
   ax1.set_title(f'Region Homogenea (Centro: Y={c_y}, X={c_x})')
   ax1.axis('off')
    
   # --- Subplot 2: Histograma y CDF superpuestos ---
   ax2.bar(range(256), hist, color='gray', label='Histograma', width=1.0)
   ax2.set_xlabel('Nivel de Gris')
   ax2.set_ylabel('Cantidad de Pixeles')
    
   # Crear un segundo eje Y (derecho) para la CDF porque tiene distinta escala (0 a 1)
   ax3 = ax2.twinx()
   ax3.plot(cdf, color='red', linewidth=2, label='CDF')
   ax3.set_ylabel('Probabilidad Acumulada (CDF)')
   ax3.set_ylim([0, 1.05])
    
    
   # Juntar leyendas
   lines_1, labels_1 = ax2.get_legend_handles_labels()
   lines_2, labels_2 = ax3.get_legend_handles_labels()
   ax3.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
    
   plt.show()

   

def transformacion(img, distancia_y, distancia_x, alto_region: int, ancho_region: int, control_limite: float):
   m = np.zeros_like(img)
   centros_y, centros_x = malla(img, distancia_y, distancia_x)
   cdf_img = {}
   # obtencion de los centros junto a su cdf
   for c_y in centros_y:
      for c_x in centros_x:
         region = extraer_region(img, alto_region, ancho_region, c_y, c_x)
         if control_limite is None:
            cdf_img[(c_y, c_x)] = calculo_cdf(region)
         else:
            cdf_img[(c_y, c_x)] = contraste(region,control_limite)
   # ver donde caen los pixeles con relacion a los centros
   for y in range(len(img)):
      for x in range(len(img[y])):
       
       
       v = img[y,x]

       # check centros en y
       if y <= centros_y[0]:

        y1 = centros_y[0]
        y2 = centros_y[0]
       elif y >= centros_y[-1]:
        y1 = centros_y[-1]
        y2 = centros_y[-1]

       else:
          for i in range(1, len(centros_y)):
            if y <= centros_y[i]:
                y1 = centros_y[i-1]
                y2 = centros_y[i]
                break
    
          
       # check centros en x
       if x <= centros_x[0]:
          x1 = centros_x[0]
          x2 = centros_x[0]
       elif x >= centros_x[-1]:
          x1 = centros_x[-1]
          x2 = centros_x[-1]
       else:     
          for i in range(1, len(centros_x)):
              if x <= centros_x[i]:
                x1 = centros_x[i-1]
                x2 = centros_x[i]
                break
           
          

      #para evitar dividir por cero

       if y1 == y2:
        peso_y1 = 1.0
        peso_y2 = 0.0
       else:
        peso_y1 = (y2-y)/(y2-y1)
        peso_y2 = (y-y1)/(y2-y1)
       if x1 == x2:
        peso_x1 = 1.0
        peso_x2 = 0.0
       else:
        peso_x1 = (x2-x)/(x2-x1)
        peso_x2 = (x-x1)/(x2-x1)

      # esto es para la eliminar pesos
      #  peso_y1 = 1.0
      #  peso_y2 = 0.0
      #  peso_x1 = 1.0
      #  peso_x2 = 0.0
       
       v11 = cdf_img[(y1,x1)][v]
       v12 = cdf_img[(y1,x2)][v]
       v21 = cdf_img[(y2,x1)][v]
       v22 = cdf_img[(y2,x2)][v]

        #suma ponderada de los pesos (interpolacion bilineal)
    
       v_resultante = (v11*peso_y1*peso_x1)+(v12*peso_y1*peso_x2) + (v21*peso_y2*peso_x1) + (v22*peso_y2*peso_x2)

       m[y,x] = v_resultante

   return m

def transformacion_bin(img, distancia_y, distancia_x, alto_region: int, ancho_region: int, control_limite: float, n_bins):
   m = np.zeros_like(img)
   centros_y, centros_x = malla(img, distancia_y, distancia_x)
   cdf_img = {}
   # obtencion de los centros junto a su cdf
   for c_y in centros_y:
      for c_x in centros_x:
         region = extraer_region(img, alto_region, ancho_region, c_y, c_x)
         if control_limite == 0.0:
            cdf_img[(c_y, c_x)] = calculo_cdf_bin(region, n_bins)

         else:
            cdf_img[(c_y, c_x)] = contraste(region,control_limite)
   # ver donde caen los pixeles con relacion a los centros
   for y in range(len(img)):
      for x in range(len(img[y])):
       
       
       v = img[y,x].astype(np.uint32)

       # check centros en y
       if y <= centros_y[0]:

        y1 = centros_y[0]
        y2 = centros_y[0]
       elif y >= centros_y[-1]:
        y1 = centros_y[-1]
        y2 = centros_y[-1]

       else:
          for i in range(1, len(centros_y)):
            if y <= centros_y[i]:
                y1 = centros_y[i-1]
                y2 = centros_y[i]
                break
    
          
       # check centros en x
       if x <= centros_x[0]:
          x1 = centros_x[0]
          x2 = centros_x[0]
       elif x >= centros_x[-1]:
          x1 = centros_x[-1]
          x2 = centros_x[-1]
       else:     
          for i in range(1, len(centros_x)):
              if x <= centros_x[i]:
                x1 = centros_x[i-1]
                x2 = centros_x[i]
                break
           
          

      #para evitar dividir por cero

       if y1 == y2:
        peso_y1 = 1.0
        peso_y2 = 0.0
       else:
        peso_y1 = (y2-y)/(y2-y1)
        peso_y2 = (y-y1)/(y2-y1)
       if x1 == x2:
        peso_x1 = 1.0
        peso_x2 = 0.0
       else:
        peso_x1 = (x2-x)/(x2-x1)
        peso_x2 = (x-x1)/(x2-x1)

       # mapear el nuevo bin original

       v = (v*n_bins) // 256
       v11 = cdf_img[(y1,x1)][v]
       v12 = cdf_img[(y1,x2)][v]
       v21 = cdf_img[(y2,x1)][v]
       v22 = cdf_img[(y2,x2)][v]

        #suma ponderada de los pesos (interpolacion bilineal)
    
       v_resultante = (v11*peso_y1*peso_x1)+(v12*peso_y1*peso_x2) + (v21*peso_y2*peso_x1) + (v22*peso_y2*peso_x2)

       m[y,x] = v_resultante

   return m



# convertir la imagen a LAB y luego pasarla a RGB?

def mostrar_imagen(img: np.ndarray):
    import matplotlib.pyplot as plt
    # # Si queremos mostrala
    plt.figure(figsize=(15,8))
    plt.imshow(img, cmap='gray') #para que se muestren los grises
    #plt.imshow(img)
    plt.show()


def clahe(img, clipLimit: float, tileGridSize: tuple):
   clahe_cv2 = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
   clahe_processed_img_cv2 = clahe_cv2.apply(img)
   return clahe_processed_img_cv2