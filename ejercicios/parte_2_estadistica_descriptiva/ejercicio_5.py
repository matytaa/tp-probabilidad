from ejercicios.parte_1_simulacion.binomial import obtener_muestra_binomial
from ejercicios.parte_2_estadistica_descriptiva.funcion_de_distribucion_empirica import funcion_de_distribución_empirica_n
from ejercicios.parte_2_estadistica_descriptiva.muestreo_de_bootstrap import generar_muestra_de_boostrap
from funciones.funciones import media_muestral
from funciones.funciones import varianza_muestral
from histogramas.histograma import frecuencia_relativa

#Repetir el experimento de los dos puntos anteriores con dos muestras aleatorias más generadas
# con los mismos parámetros. ¿Qué conclusión saca?

def repeticion_ejercicios_3_y_4():
    casos = 200
    n = 10
    p = 0.3

    array_de_valores = obtener_muestra_binomial(casos, n, p)
    distribucion_empirica = funcion_de_distribución_empirica_n(array_de_valores)
    distribucion_empirica = distribucion_empirica[::-1]

    muestra = generar_muestra_de_boostrap(distribucion_empirica, casos)

    print("media muestral = ", media_muestral(muestra))

    print("varianza muestral = ", varianza_muestral(muestra))

    frecuencia_relativa(muestra, 0.1, titulo = 'Histograma')


repeticion_ejercicios_3_y_4()

# 2do conjunto de casos
repeticion_ejercicios_3_y_4()

# CONCLUSIÓN: acorde al tamaño de la muestra con el método de boostrap denotamos cual sería la
# media poblacional y la varianza poblacional de la muestra al las medias y varianza teorica


