import numpy as np
import math
## Ejercicio 3
#Utilizando el procedimiento descrito en el capítulo 6 del Dekking (método de la función inversa o de Monte Carlo),
# implementar una función que permita generar un número aleatorio con distribución Exp(λ).

# Metodo de funcion inversa: pagina 74 del Dekkings

def fn_inversa_exponencial(_lambda, u):
    return -(1/_lambda)* math.log10(u)

def fn_exponencial_random(_lambda):
    numeroRandomConDistribucionUniforme = np.random.uniform(0,1)
    result = fn_inversa_exponencial(_lambda, numeroRandomConDistribucionUniforme)
    return result


val = fn_exponencial_random(0.5)
print(val)