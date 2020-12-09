import numpy as np
from ejercicios.parte_1_simulacion.bernuolli import muestra_bernoulli_random

# # Ejercicio 2
# 2. Utilizando la función del punto anterior, implemente otra que genere un número binomial con los parámetros n,p.

def binomial_random(n, p):
    array_valores = muestra_bernoulli_random(n, p)
    exitos = [ensayo == 1 for ensayo in array_valores]  # Cuento los exitos de los Bernoulli anteriores
    return sum(exitos)

# Implemento funcion para retornar un conjunto de valores aleatorios siguiendo la distribucion binomial
def obtener_muestra_binomial(casos, n, p):
    valores = np.zeros(casos)
    for i in range(0, casos):
        valores[i - 1] = binomial_random(n, p)
    return valores

def esperanza(n, p):
    return n*p

def varianza(n, p):
    return n*p*(1-p)


print("INICIO -> EJEMPLO DE FUNCIONAMIENTO \n")
probabilidad_de_exito = 0.4
tamanio_de_la_muestra = 100
ensayos_bernoulli = 6
exitos = binomial_random(tamanio_de_la_muestra, probabilidad_de_exito)
array_binomial = obtener_muestra_binomial(tamanio_de_la_muestra, ensayos_bernoulli, probabilidad_de_exito)
print(array_binomial)
print("Tamaño de la muestra:" + str(tamanio_de_la_muestra))
print("Numero de éxitos:" + str(exitos))
print("Probabilidad de éxito:" + str(probabilidad_de_exito))
print("FIN -> EJEMPLO DE FUNCIONAMIENTO \n")