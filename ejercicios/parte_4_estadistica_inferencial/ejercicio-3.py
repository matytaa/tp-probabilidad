import math
import scipy.stats as stats
from ejercicios.parte_1_simulacion.normal import fn_normal_array

## Ejercicio 4.3
#3. Repita el punto anterior pero usando la varianza estimada s^2 , para la muestra de tamaño adecuado.

def obtener_s(valores, x_raya, n):
    sumatoria = 0
    for i in range (0, len(valores)):
        sumatoria += math.pow(valores[i]-x_raya,2)
    return sumatoria/(n-1)

def con_varianza_desconocida(n, confianza, valores, mu):
    x_raya = mu
    alfa = 1 - confianza
    alfa_sobre_2 = alfa / 2
    s = obtener_s(valores, x_raya, n)
    t_alfa_sobre_2 = stats.t.ppf(alfa_sobre_2, n-1)
    limite_inferior = x_raya + (t_alfa_sobre_2*(s/math.sqrt(n)))
    limite_superior = x_raya - (t_alfa_sobre_2*(s/math.sqrt(n)))
    return limite_inferior, limite_superior


confianza = 0.95
n = 10
valores = fn_normal_array(10,100, 5)
mu = 100
limite_inferior, limite_superior = con_varianza_desconocida(n, confianza, valores, mu)
print("Intervalo de confianza, con varianza conocida, al 95% se encuentra entre: "+ str(limite_inferior) + " <= U <= "+str(limite_superior))

confianza = 0.98
limite_inferior, limite_superior = con_varianza_desconocida(n, confianza, valores, mu)
print("Intervalo de confianza, con varianza conocida, al 98% se encuentra entre: "+ str(limite_inferior) + " <= U <= "+str(limite_superior))