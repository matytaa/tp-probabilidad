import numpy as np


def generar_muestra_segun_cdf(distribucion_empirica, casos):
    muestra = []
    for i in range(0, casos):
        muestra.append(np.random.uniform(0, 1))

    muestra_nueva = []
    for i in range(0, casos):
        valor = muestra[i]
        for j in (range(0, len(distribucion_empirica))):
            if (distribucion_empirica[j][1] >= valor):
                muestra_nueva.append(distribucion_empirica[j][0])
                break
    return muestra_nueva
