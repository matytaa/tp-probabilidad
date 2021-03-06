import collections
from ejercicios.parte_1_simulacion.binomial import obtener_muestra_binomial
from ejercicios.parte_2_estadistica_descriptiva.funcion_de_distribucion_empirica import funcion_de_distribución_empirica_n
from ejercicios.parte_2_estadistica_descriptiva.funcion_de_distribucion_empirica import graficar_diagrama_acumulada
from ejercicios.parte_2_estadistica_descriptiva.muestreo_de_bootstrap import generar_muestra_de_boostrap
from funciones.funciones import media_muestral
from funciones.funciones import varianza_muestral
from histogramas.histograma import frecuencia_relativa


## Parte 2 Ejercicio 3
# Generar una muestra de números Bin(10, 0.3) de tamaño de muestra N = 50.
# Construir la función de distribución empírica de dicha muestra.3
casos = 50
n = 10
p = 0.3
muestras_binomiales = obtener_muestra_binomial(casos, n, p)
distribucion_empirica = funcion_de_distribución_empirica_n(muestras_binomiales)
print("muestrar binomial: " + str(sorted(muestras_binomiales)))
print("F de distribucion empirica: ", distribucion_empirica)
graficar_diagrama_acumulada(distribucion_empirica)

## Parte 2 Ejercicio 4
# A partir de la función de distribución empírica del punto anterior,
# generar una nueva muestra de números aleatorios utilizando
# el método de simulación de la primera parte.
# Computar la media y varianza muestral y graficar el histograma.

# Ejemplos de boostrap:
# https://datasciencechalktalk.com/2019/11/12/bootstrap-sampling-an-implementation-with-python/
distribucion_empirica = distribucion_empirica[::-1]

muestra_de_bootstrap = generar_muestra_de_boostrap(distribucion_empirica, casos)

print ("Número + Cantidad de ocurrencias encontradas ==> %s" % collections.Counter(sorted(muestra_de_bootstrap)))
print("\n")

muestra = generar_muestra_de_boostrap(distribucion_empirica, casos)
array_empirica = []
for i in range(0, len(distribucion_empirica)):
    array_empirica.append(distribucion_empirica[i][1])
print("media array_empirica = ", media_muestral(array_empirica))
print("media muestral = ", media_muestral(muestra_de_bootstrap))

print("varianza array_empirica = ", varianza_muestral(array_empirica))
print("varianza muestral = ", varianza_muestral(muestra_de_bootstrap))

frecuencia_relativa(muestra_de_bootstrap, ancho_de_barra = 0.1, titulo = 'Histograma')
