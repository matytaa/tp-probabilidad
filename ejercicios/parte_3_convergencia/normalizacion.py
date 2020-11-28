import numpy as np
import pandas as pd

# Para normalizar el valor, tomo la siguiente normalización = ( x – min(x) ) / ( max(x) – min(x) )
def normalizar(valor_a_normalizar):
    minimo = (min(valor_a_normalizar))
    maximo = (max(valor_a_normalizar))

    calculo_b = maximo - minimo

    normal = np.zeros((200))

    for i in range(0, 200):
        valor = valor_a_normalizar[i - 1]
        calculo_a = valor - minimo
        calculo = calculo_a / calculo_b
        normal[i - 1] = calculo

    print("Normal: "+ str(normal))
    df = pd.DataFrame({"valor": valor_a_normalizar, "normalizado": normal})
    print(df)
    return normal