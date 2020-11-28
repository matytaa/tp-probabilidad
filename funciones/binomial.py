import numpy as np
import matplotlib.pyplot as plt
import collections

# # Ejercicio 2
# 2. Utilizando la función del punto anterior, implemente otra que genere un número binomial con los parámetros n,p.

def fn_binomial_random(n, p):
    intentos = [np.random.uniform(0, 1) for x in range(0, n)]
    exitos = [intento <= p for intento in intentos]
    return sum(exitos)

# Implemento funcion para retornar un conjunto de valores aleatorios siguiendo la distribucion binomial
def fn_binomial_array(casos, n, p):
    valores = np.zeros(casos)
    for i in range(0, casos):
        valores[i - 1] = fn_binomial_random(n, p)
    return valores

def graficaBinomial(puntostotales, n, p):
    Xs = [k / n for k in range(0, n + 1)]
    Ys = [0 for i in range(0, n + 1)]
    puntoactual = 0
    while puntoactual < puntostotales:
        ubicacion = fn_binomial_random(n, p)
        Ys[ubicacion] += 1
        puntoactual += 1
    return Xs, Ys

def esperanza(n, p):
    return n*p

def varianza(n, p):
    return n*p*(1-p)

val = fn_binomial_random(50, 0.4)
print(val)
val = fn_binomial_array(1000, 6, 0.5)
print(val)
collections.Counter(val)

ns = [6]
p = 0.5
puntos = 1000
for n in ns:
    curva = graficaBinomial(puntos, n, p)
    plt.plot(*curva, label=f'n = {n}')
plt.xlabel('Probabilidad ')
plt.ylabel('')
plt.legend()
plt.show()