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


val = fn_gaussian_random(2, 3)
print(val)

val = fn_normal_array(10,100, 5)
print(val)