import numpy as np
import matplotlib.pyplot as plt


## Generar una muestra de números Bin(10, 0.3) de tamaño de muestra N = 50. Construir la función de distribución empírica de dicha muestra.
# Referencia Funcion de Distribucion Empirica
# https://machinelearningmastery.com/empirical-distribution-function-in-python/#:~:text=An%20empirical%20distribution%20function%20can,specific%20observations%20from%20the%20domain.
# Referencia sobre Funcion de Distribucion Empirica
# http://halweb.uc3m.es/esp/Personal/personas/jmmarin/esp/Boots/tema2BooPres.pdf

def funcion_de_distribución_empirica_n(muestras):
    muestras = sorted(muestras)
    mapa_de_probabilidad_por_cada_valor = np.zeros((len(muestras), 2))

    cantidad = len(muestras)
    valor_maximo = max(muestras)
    tamanio_de_la_muestra = 0

    for i in range(0, cantidad):
        if (muestras[i] < valor_maximo):
            tamanio_de_la_muestra = tamanio_de_la_muestra + 1

    for i in range(0, tamanio_de_la_muestra):
        mapa_de_probabilidad_por_cada_valor[i, 0] = muestras[i]
        mapa_de_probabilidad_por_cada_valor[i, 1] = round((i + 1) / (tamanio_de_la_muestra + 1), 2)

    for i in range(tamanio_de_la_muestra, cantidad):
        mapa_de_probabilidad_por_cada_valor[i, 0] = muestras[i]
        mapa_de_probabilidad_por_cada_valor[i, 1] = 1

    return mapa_de_probabilidad_por_cada_valor


def graficar_diagrama_acumulada(valores):
    x, y = zip(*valores)
    plt.step(x,y)
    plt.title('Probabilidad Acumulada F Empirica')
    plt.grid()
    plt.show()

