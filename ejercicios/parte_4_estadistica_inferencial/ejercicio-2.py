import math
import scipy.stats as stats
from ejercicios.parte_1_simulacion.normal import fn_normal_array

## Ejercicio 4.2
# 2. Suponga que ya conoce el dato de que la distribución tiene varianza 5.
# Obtener intervalos de confianza del 95% y 98% para la media de ambas muestras.

def con_varianza_conocida(varianza, n, confianza, mu):
    sigma = math.sqrt(varianza)
    alfa = 1 - confianza
    alfa_sobre_2 = alfa / 2
    z_alfa_sobre_2 = stats.norm.ppf(alfa_sobre_2)
    x_raya = mu
    limite_inferior = x_raya + (z_alfa_sobre_2*(sigma/math.sqrt(n)))
    limite_superior = x_raya - (z_alfa_sobre_2*(sigma/math.sqrt(n)))
    return limite_inferior, limite_superior


mu = 100
varianza = 5
n = 10

confianza = 0.95
limite_inferior, limite_superior = con_varianza_conocida(varianza, n, confianza, mu)
print("Intervalo de confianza, con varianza conocida, al 95% se encuentra entre: "+ str(limite_inferior) + " <= U <= "+str(limite_superior))

confianza = 0.98
limite_inferior, limite_superior = con_varianza_conocida(varianza, n, confianza, mu)
print("Intervalo de confianza, con varianza conocida, al 98% se encuentra entre: "+ str(limite_inferior) + " <= U <= "+str(limite_superior))