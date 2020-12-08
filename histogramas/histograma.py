import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import collections

def graficar_frecuencia_relativa(x, res, titulo):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(1, 1, 1)

    ax.bar(x, res.frequency, width=res.binsize)
    ax.set_title(titulo)
    ax.set_xlim([x.min(), x.max()])

    plt.show()

def frecuencia_relativa_con_ancho(conjuntoDeDatos, ancho=0.1, titulo='Histograma de frecuencia relativa'):
    inicio = int(min(conjuntoDeDatos))
    fin = int(max(conjuntoDeDatos))
    n = int((fin - inicio) / (ancho))
    res = stats.relfreq(conjuntoDeDatos, numbins=n)
    x = res.lowerlimit + np.linspace(0, res.binsize * res.frequency.size, res.frequency.size)
    graficar_frecuencia_relativa(x, res, titulo)


def frecuencia_relativa(conjuntoDeDatos, ancho_de_barra=0.1, titulo='Histograma de frecuencia relativa'):
    repeticiones = collections.Counter(conjuntoDeDatos)

    total = len(conjuntoDeDatos)
    valores = []
    x = []
    suma = 0
    i = 0
    for clave, valor in repeticiones.items():
        valores.append(valor / total)
        x.append(clave)
        suma += valores[i]
        i += 1
    plt.title(titulo)
    plt.bar(x, valores, ancho_de_barra)
    plt.show()