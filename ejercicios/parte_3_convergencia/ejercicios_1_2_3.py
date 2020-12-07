import math
import numpy as np
import pandas as pd
from funciones.funciones import media_muestral
from funciones.funciones import varianza_muestral
from histogramas.histograma import frecuencia_relativa
from ejercicios.parte_1_simulacion.binomial import obtener_muestra_binomial
from ejercicios.parte_3_convergencia.normalizacion import normalizar

## Parte 3: Convergencia

# # Parte 3 Ejercicio 1
# Generar cuatro muestras de números aleatorios de tamaño 100, todas con distribución binomial con p = 0,40 y n = 10, n = 20,
# n = 50 y n = 100 respectivamente. Graficar sus histogramas. ¿Qué observa?

# CONCLUSION: Teniendo en cuenta el p=0.4, puedo estimar que el pico de frencuencia en cada uno, va a estar determinado por el
# valor del rango que sea la esperanza (n*p) aproximadamente.

val_e1_p3_10 = obtener_muestra_binomial(100, 10, 0.4)
print(val_e1_p3_10)
frecuencia_relativa(val_e1_p3_10)


val_e1_p3_20 = obtener_muestra_binomial(100, 20, 0.4)
print(val_e1_p3_20)
frecuencia_relativa(val_e1_p3_20)


val_e1_p3_50 = obtener_muestra_binomial(100, 50, 0.4)
print(val_e1_p3_50)
frecuencia_relativa(val_e1_p3_50)

val_e1_p3_100 = obtener_muestra_binomial(100, 100, 0.4)
print(val_e1_p3_100)
frecuencia_relativa(val_e1_p3_100)



# #  Parte 3 Ejercicio 2
# Elija la muestra de tamaño 200 y calcule la media y desviación estándar muestral. Luego, normalice cada dato de la muestra
# y grafique el histograma de la muestra normalizada. Justifique lo que observa.

val_e2_p3_200 = obtener_muestra_binomial(200, 10, 0.4)
val_e2_p3_200 = sorted(val_e2_p3_200)

print(val_e2_p3_200)

media_muestral_p3_2 = media_muestral(val_e2_p3_200)
print("media muestral: ",media_muestral_p3_2)

varianza_p3_200 = varianza_muestral(val_e2_p3_200)
desviacion_estandar = math.sqrt(varianza_p3_200)
print("desviacion estandar:", desviacion_estandar)

normal = normalizar(val_e2_p3_200, media_muestral_p3_2, varianza_p3_200)
frecuencia_relativa(normal)
# En el primer conjunto, la esperanza de la binomial, me da que en el valor 4 tendre el pico de frencuencia de ocurrencia
# Cuando normalizo los valores a un intervalo {0,1}, el valor representado de 4 es 0,5 esto en el histograma
# me muestra el valor con mayor frecuencia de ocurrencia.



##  Parte 3 Ejercicio 3
# Para cada una de las muestras anteriores, calcule la media muestral. Justifique lo que observa.
print("media muestral ejercicio 1 N=100 n=10 = ",media_muestral(val_e1_p3_10))
print("media muestral ejercicio 1 N=100 n=20 = ",media_muestral(val_e1_p3_20))
print("media muestral ejercicio 1 N=100 n=50 = ",media_muestral(val_e1_p3_50))

print("media muestral ejercicio 1 N=100 n=100= ",media_muestral(val_e1_p3_100))
print("Esperanza n=100 = ",0.4*100)

print("media muestral ejercicio 2 N=200 n=10 = ",media_muestral(val_e2_p3_200))
print("Esperanza n=10 = ",0.4*10)

# Al modificar el parametro n de las distribuciones, estoy modificando la cantidad de éxitos que voy
# a tener con probabilidad 0.4
# Ademas, en el último caso, aproximo mejor el valor de esperanza de binomial, respecto al
# primero, ya que estoy tomando mas casos.
