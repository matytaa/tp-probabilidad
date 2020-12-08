from ejercicios.parte_1_simulacion.binomial import obtener_muestra_binomial
from ejercicios.parte_2_estadistica_descriptiva.funcion_de_distribucion_empirica import funcion_de_distribución_empirica_n
from ejercicios.parte_2_estadistica_descriptiva.muestreo_de_bootstrap import sampleo_bootstrap
from funciones.funciones import media_muestral
from funciones.funciones import varianza_muestral
from histogramas.histograma import frecuencia_relativa

def repeticion_ejercicios_3_y_4():
    casos = 50
    n = 10
    p = 0.3

    array_de_valores = obtener_muestra_binomial(casos, n, p)
    distribucion_empirica = funcion_de_distribución_empirica_n(array_de_valores)
    distribucion_empirica = distribucion_empirica[::-1]

    muestra = sampleo_bootstrap(distribucion_empirica, casos)

    print("media muestral array de valores 1 = ",
          media_muestral(array_de_valores))
    print("media muestral array de valores 2 = ",
          media_muestral(muestra))

    print("varianza muestral array de valores 1 = ",
          varianza_muestral(array_de_valores))
    print("varianza muestral array de valores 2 = ",
          varianza_muestral(muestra))

    frecuencia_relativa(muestra, 0.1, titulo = 'Histograma de nueva muestra')


repeticion_ejercicios_3_y_4()

# 2do conjunto de casos
repeticion_ejercicios_3_y_4()

# CONCLUSIÓN: Por un lado, la función escalonada de distribución empirica me muestra los saltos que tendrán cada
# valor de la muestra respecto de su probabilidad asignada como i/n con i subindice de la muestra y n cantidad de valores.
# Por otro lado, respecto al muestreo de Bootstrap, con este obtenemos conjuntos de datos que representarán la media de cada submuestra.
# Teniendo en cuenta, que la muestra origianl obtenida de forma aleatoria por la distribución, pasará a ser la población.
# Conforme aumentamos la cantidad de casos generados por el muestreo, este aproximará al grafico de una normal y
# los valores obtenidos (que son medias), contendrán al valor de la media muestral original.
# Ademas, observamos que la varianza muestral en la muestra de bootstrap, es mucho mas chica que en el conjunto original.
# Por Teorema Central del Limite, como tiene n>30, y es muy grande, la grafica se asemeja a una distribucion normal.

