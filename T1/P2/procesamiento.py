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
  plt.figure(figsize=(60,9))
  plt.bar(range(n),h[0:n])
  plt.show()



def hist_forceuni(img):
    # input: img
    # output: equalized image y

    img_x = img.copy()
    img_x = 255 * ((img_x - img_x.min()) / (img_x.max() - img_x.min()))

    n, m = img_x.shape
    y = np.zeros((n * m, 1), dtype=np.uint8)
    j = np.argsort(img_x.flatten())
    z = np.zeros((n * m, 1), dtype=np.uint8)
    d = int(np.fix((n * m / 256) + 0.5))

    for i in range(255):
        z[i * d:(i + 1) * d] = i * np.ones((d, 1))  # , dtype=np.uint8)
    z[255 * d:n * m] = 255 * np.ones((n * m - 255 * d, 1))  # , dtype=np.uint8)

    y[j] = z
    y = y.reshape(n, m)

    return y


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
    

def transformacion(img, distancia_y, distancia_x, alto_region: float, ancho_region: float):
   m = np.zeros_like(img)
   centros_y, centros_x = malla(img, distancia_y, distancia_x)
   cdf_img = {}
   # obtencion de los centros junto a su cdf
   for c_y in centros_y:
      for c_x in centros_x:
         region = extraer_region(img, alto_region, ancho_region, c_y, c_x)
         cdf_img[(c_y, c_x)] = calculo_cdf(region)
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
       
       v11 = cdf_img[(y1,x1)][v]
       v12 = cdf_img[(y1,x2)][v]
       v21 = cdf_img[(y2,x1)][v]
       v22 = cdf_img[(y2,x2)][v]

        #suma ponderada de los pesos (interpolacion bilineal)

       v_resultante = (v11*peso_y1*peso_x1)+(v12*peso_y1*peso_x2) + (v21*peso_y2*peso_x1) + (v22*peso_y2*peso_x2)

       m[y,x] = v_resultante

   return m

    
def contraste():
   pass




img = image("P2_IMG_2423.tif", os.getcwd())

centros_y, centros_x = malla(img, 50, 50)

r1 = extraer_region(img,100, 100, centros_y[0],centros_x[0])


# convertir la imagen a LAB y luego pasarla a RGB?

def mostrar_imagen(img: np.ndarray):
    import matplotlib.pyplot as plt
    # # Si queremos mostrala
    plt.figure(figsize=(15,8))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255) #para que se muestren los grises
    #plt.imshow(img)
    plt.show()
