import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
## Generar una muestra de números Bin(10, 0.3) de tamaño de muestra N = 50.
# Construir la función de distribución empírica de dicha muestra.

def funcion_de_distribución_empirica_n(muestras):
    muestras = sorted(muestras)
    mapa_de_probabilidad_por_cada_valor = defaultdict(list)

    cantidad = len(muestras)
    valor_maximo = max(muestras)
    tamanio_de_la_muestra = 0

    for i in range(0, cantidad):
        if (muestras[i] < valor_maximo):
            tamanio_de_la_muestra = tamanio_de_la_muestra + 1

    for i in range(0, tamanio_de_la_muestra):
        mapa_de_probabilidad_por_cada_valor[muestras[i]] = round((i + 1) / (tamanio_de_la_muestra + 1), 2)

    for i in range(tamanio_de_la_muestra, cantidad):
        mapa_de_probabilidad_por_cada_valor[muestras[i]] = 1

    funcion_de_distribucion = np.zeros((len(mapa_de_probabilidad_por_cada_valor), 2))
    for i in range(0, len(mapa_de_probabilidad_por_cada_valor)):
        clave, valor = mapa_de_probabilidad_por_cada_valor.popitem()
        funcion_de_distribucion[i, 0] = clave
        funcion_de_distribucion[i, 1] = valor

    return funcion_de_distribucion


def graficar_diagrama_acumulada(valores):
    x, y = zip(*valores)
    plt.step(x,y)
    plt.title('Probabilidad Acumulada F Empirica')
    plt.grid()
    plt.show()

