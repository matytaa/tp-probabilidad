import numpy as np
import math
import collections
from ejercicios.parte_1_simulacion.exponencial import fn_exponencial_random
from histogramas.histograma import graficar_histograma
# # Parte 2: Estadística descriptiva
#Ahora vamos a aplicar las técnicas vistas en la materia al estudio de algunas muestras de datos.

# # Ejercicio 1
#Generar tres muestras de números aleatorios Exp(0, 5) de tamaño n = 10, n = 30 y n = 200.
#Para cada una, computar la media y varianza muestral.
# ¿Qué observa?

def fn_generador_de_muestras_numeros_random_con_dist_exponencial(n, _lambda):
    valores = np.zeros(n)
    for i in range(0, n):
        valores[i - 1] = fn_exponencial_random(_lambda)

    # Calculo de esperanza
    suma = 0
    for i in range(0, 10):
        suma = suma + valores[i]
    esperanza = suma / 10

    # Calculo de varianza
    suma = 0
    for i in range(0, 10):
        suma = suma + math.pow(valores[i], 2)
    varianza = suma / 10 - math.pow(esperanza, 2)

    return valores, esperanza, varianza


# Primera muestra con n=10:
valores_n10, esperanza, varianza = fn_generador_de_muestras_numeros_random_con_dist_exponencial(10, 0.5)
print(valores_n10)
collections.Counter(valores_n10)
print("Media=", esperanza)
print("Varianza=", varianza)
graficar_histograma(valores_n10, 10, 0.4)
graficar_histograma(valores_n10, 10, 0.2)
graficar_histograma(valores_n10, 10, 0.1)


# Segunda muestra con n=30:
valores_n30, esperanza, varianza = fn_generador_de_muestras_numeros_random_con_dist_exponencial(30, 0.5)
print(valores_n30)
collections.Counter(valores_n30)
print("Media=", esperanza)
print("Varianza=", varianza)
graficar_histograma(valores_n30,30,0.4)
graficar_histograma(valores_n30,30,0.2)
graficar_histograma(valores_n30,30,0.1)


# tercera muestra con n=200:
valores_n200, esperanza, varianza = fn_generador_de_muestras_numeros_random_con_dist_exponencial(200, 0.5)
print(valores_n200)
collections.Counter(valores_n200)
print("Media=", esperanza)
print("Varianza=", varianza)
graficar_histograma(valores_n200, 200, 0.4)
graficar_histograma(valores_n200, 200, 0.2)
graficar_histograma(valores_n200, 200, 0.1)
