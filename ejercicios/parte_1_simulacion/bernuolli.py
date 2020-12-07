import numpy as np
import collections

## Ejercicio 1
# Utilizando únicamente la función random de su lenguaje (la función que genera un número aleatorio uniforme entre 0 y 1),
# implemente una función que genere un número distribuido Bernoulli con probabilidad p.

# La Distribución de Bernoulli describe un experimento probabilístico en donde el ensayo
# tiene dos posibles resultados, éxito o fracaso.

# p   es la probabilidad de éxito
# 1−p es la probabilidad de fracaso

# np.random.uniform(0,1) "la función que genera un número aleatorio uniforme entre 0 y 1"
# De esta forma, devuelve valores equiprobables entre 0 y 1
# Los valores, los voy a generar de esta forma valor = np.random.uniform(0,1) y dps, comparo con el valor de p (prob)

def fn_bernoulli_random(p):
    if np.random.uniform(0, 1) > p:
        return 0
    else:
        return 1

def muestra_bernoulli_random(x, p):
    valores = np.zeros((x))
    for i in range(0, x):
        valores[i - 1] = fn_bernoulli_random(p)
    return valores

prueba_bernoulli = muestra_bernoulli_random(100, 0.4)
print(prueba_bernoulli)
print("Comprobacion, agrupacion por valores ==> %s" % collections.Counter(prueba_bernoulli))