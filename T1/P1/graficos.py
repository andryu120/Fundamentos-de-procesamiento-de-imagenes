
import numpy as np
from main import puntos

def graficar(punto,titulo):
    import matplotlib.pyplot as plt
    # desernollamos el circ
    h_inicial, m_inicial = punto[0]
    p_ext = punto + [(h_inicial+360,m_inicial)]

    h_vals = [punto[0] for punto in p_ext]
    m_vals = [punto[1] for punto in p_ext]

    plt.figure(figsize = (10,4))
    plt.plot(h_vals,m_vals, marker='o', linestyle='-', color='b', linewidth=2)

    plt.title(titulo)
    plt.xlabel("hue (h)")
    plt.ylabel("Amplificacion (m)")
    plt.grid(True)
    plt.legend()
    plt.show()

for titulo, p in puntos.items():
    graficar(p, titulo)
