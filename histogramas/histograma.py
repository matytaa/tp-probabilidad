import numpy as np
import matplotlib.pyplot as plt

def graficar_histograma(valor, muestra, ancho):
    inicio = int(min(valor))
    fin = int(max(valor))
    print("Inicio=", inicio)
    print("Fin=", fin)
    ancho = 0.4
    div = np.linspace(inicio, fin, round(1 + (fin - inicio) / ancho))
    plt.figure("Valores con ancho " + str(ancho) + ", muestra n =" + str(muestra))
    plt.hist(valor, div)
    plt.grid()
    plt.show()