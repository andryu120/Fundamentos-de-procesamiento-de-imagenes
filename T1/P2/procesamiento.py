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
    Xbgr = cv2.imread(name,cv2.IMREAD_UNCHANGED)
    imagen = cv2.cvtColor(Xbgr, cv2.COLOR_BGR2RGB) # conversion de BGR a RGB

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


def division_malla(size: float, img):
   pass


def calculo_local():
   pass

def interpolacion():
   pass


# convertir la imagen a LAB y luego pasarla a RGB