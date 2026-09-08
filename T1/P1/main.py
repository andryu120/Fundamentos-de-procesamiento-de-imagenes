
import p1 as f
import os
path = os.getcwd()
p = [(0, 0.0), (120, 2.0), (240, 2.0), (350, 0.0)]
if __name__ == "__main__":
    modo = str(input())
    if modo == ("HSV" or "hsv"):
        
        imagen_original, imagen = f.image("imagen.png", path)
        imagen_hsv = f.rgb_to_hsv(imagen)
        p = f.correcion_m(p)
        M = f.interpolar(imagen_hsv, p)
        S_prima = f.transformacion(imagen_hsv,M)
        imagen_rgb = f.hsv_to_rgb(imagen_hsv, S_prima)
        f.mostrar_imagen(imagen_rgb)
    elif modo == ("CIE" or "cie" or " CIE L*c*h*"):
        pass
    else:
        raise ValueError("Modo incorrecto, ingrese de nuevo el modo")
    