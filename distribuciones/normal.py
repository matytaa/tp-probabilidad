import numpy as np
import math

# # Ejercicio 4
# Investigar como generar números aleatorios con distribución normal. Implementarlo.
# Usamos el método de Box-muller para generar numeros random siguiendo una distribucion normal
# https://es.wikipedia.org/wiki/M%C3%A9todo_de_Box-Muller

def fn_gaussian_random(mean, stddev):
    theta = 2 * math.pi * np.random.uniform(0,1)
    rho = math.sqrt(-2 * math.log10(1 - np.random.uniform(0,1)))
    scale = stddev * rho
    x = mean + scale * math.cos(theta)
    y = mean + scale * math.sin(theta)
    return y

def fn_normal_array(casos,mean,stddev):
    valores = np.zeros(casos)
    for i in range(0,casos):
        valores[i-1] = fn_gaussian_random(mean, stddev)
    return valores

def esperanza(muestra):
    sumatoria = suma_de_valores(muestra)
    return sumatoria / len(muestra)

def varianza(muestra):
    varianza = 0
    sumatoria = suma_de_valores(muestra)
    largo_de_la_muestra = len(muestra)
    for i in range(0, largo_de_la_muestra):
        varianza = varianza + (muestra[i - 1] - sumatoria) ** 2
    return varianza / largo_de_la_muestra

def suma_de_valores(muestra):
    suma_b = 0
    for i in range(0, len(muestra)):
        suma_b = suma_b + muestra[i - 1]
    return suma_b

val = fn_gaussian_random(2, 3)
print(val)

val = fn_normal_array(10,100, 5)
print(val)