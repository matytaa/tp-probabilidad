from ejercicios.parte_2_estadistica_descriptiva.generador_de_muestras_exponenciales import fn_generador_de_muestras_numeros_random_con_dist_exponencial
from histogramas.histograma import frecuencia_relativa_con_ancho

## Parte 2 Ejercicio 1
# Generar tres muestras de números aleatorios Exp(0,5) de tamaño n = 10, n = 30 y n = 200.
# Para cada una, computar la media y varianza muestral. ¿Qué observa?


# Primera muestra con n=10:
val_n10, media, varianza = fn_generador_de_muestras_numeros_random_con_dist_exponencial(10,0.5)
print("N=", 10)
print("Media=", media)
print("Varianza=", varianza)


# Segunda muestra con n=30:
val_n30, media, varianza = fn_generador_de_muestras_numeros_random_con_dist_exponencial(30,0.5)
print("N=", 30)
print("Media=", media)
print("Varianza=", varianza)


# Segunda muestra con n=200:
val_n200, media, varianza = fn_generador_de_muestras_numeros_random_con_dist_exponencial(200,0.5)
print("N=", 200)
print("Media=", media)
print("Varianza=", varianza)

# OBSERVACION: Conforme aumente la cantidad de muestra la esperanza y la varianza se deberían de estabilizar.


## Parte 2 Ejercicio 2
# Para las tres muestras anteriores, graficar los histogramas de frecuencias relativas con anchos de banda 0,4, 0,2 y 0,1; es decir,
# un total de 9 histogramas. ¿Qué conclusiones puede obtener?

#Histogramas para la muestra de n = 10
frecuencia_relativa_con_ancho(val_n10, 0.4)
frecuencia_relativa_con_ancho(val_n10, 0.2)
frecuencia_relativa_con_ancho(val_n10, 0.1)

#Histogramas para la muestra de n = 30
frecuencia_relativa_con_ancho(val_n30, 0.4)
frecuencia_relativa_con_ancho(val_n30, 0.2)
frecuencia_relativa_con_ancho(val_n30, 0.1)

#Histogramas para la muestra de n = 200
frecuencia_relativa_con_ancho(val_n200, 0.4)
frecuencia_relativa_con_ancho(val_n200, 0.2)
frecuencia_relativa_con_ancho(val_n200, 0.1)


# OBSERVACIONES: Al realizar los histogramas, conforme a la muestra aumenta el histograma refleja una exponencial.
