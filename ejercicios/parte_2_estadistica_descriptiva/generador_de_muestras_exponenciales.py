import numpy as np
from ejercicios.parte_1_simulacion.exponencial import fn_exponencial_random
from funciones.funciones import media_muestral, varianza_muestral

## Parte 2: Estadística descriptiva
#Ahora vamos a aplicar las técnicas vistas en la materia al estudio de algunas muestras de datos.

# # Ejercicio 1
#Generar tres muestras de números aleatorios Exp(0, 5) de tamaño n = 10, n = 30 y n = 200.
#Para cada una, computar la media y varianza muestral.
# ¿Qué observa?

def fn_generador_de_muestras_numeros_random_con_dist_exponencial(n, _lambda):
    valores = np.zeros(n)
    for i in range(0, n):
        valores[i] = fn_exponencial_random(_lambda)
    return valores, media_muestral(valores), varianza_muestral(valores)
