
import p1 as f
import os

path = os.getcwd()
print(path)
p = [(0, 0.0), (120, 2.0), (240, 2.0), (350, 0.0)]
if __name__ == "__main__":
    modo = str(input())
    img = f.color_saturation("imagen.png", path, p, modo)
    
    
    f.mostrar_imagen(img)
    imagen_original, copia = f.image("imagen.png",path)
    f.mostrar_imagen(imagen_original)