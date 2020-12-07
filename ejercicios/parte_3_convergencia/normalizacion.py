import numpy as np
import pandas as pd

def normalizar(muestra_a_normalizar, media_muestral, varianza):
    normal = np.zeros(len(muestra_a_normalizar))
    for i in range(0, len(muestra_a_normalizar)):
        valor = muestra_a_normalizar[i]
        calculo_a = valor - media_muestral
        calculo = calculo_a / varianza
        normal[i] = calculo

    df = pd.DataFrame({"valor": muestra_a_normalizar, "estandarizado": normal})
    print(df)
    return normal