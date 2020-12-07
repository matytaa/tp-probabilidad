import math
import numpy as np
## Ejercicio 3
#Utilizando el procedimiento descrito en el capítulo 6 del Dekking (método de la función inversa o de Monte Carlo),
# implementar una función que permita generar un número aleatorio con distribución Exp(λ).
# Metodo de funcion inversa: pagina 74 del Dekkings

def inversa_de_una_exponencial(un_lambda, u):
    return -(1 / un_lambda) * math.log(1 - u)

def fn_exponencial_random(un_lambda):
    numero_random = np.random.uniform(0, 1)
    result = inversa_de_una_exponencial(un_lambda, numero_random)
    return result

val = fn_exponencial_random(0.5)
print(val)