from funciones.binomial import fn_binomial_array
from histogramas.histograma import graficar_histograma
import collections

# # Parte 3: Convergencia
# # Ejercicio 1
# Generar cuatro muestras de números aleatorios de tamaño 100, todas con distribución binomial con p = 0,40 y n = 10, n = 20,
# n = 50 y n = 100 respectivamente. Graficar sus histogramas. ¿Qué observa?

val_e1_p3_10 = fn_binomial_array(100,10,0.4)
print(val_e1_p3_10)
collections.Counter(val_e1_p3_10)


val_e1_p3_20 = fn_binomial_array(100,20,0.4)
print(val_e1_p3_20)
collections.Counter(val_e1_p3_20)


val_e1_p3_50 = fn_binomial_array(100,50,0.4)
print(val_e1_p3_50)
collections.Counter(val_e1_p3_50)


val_e1_p3_100 = fn_binomial_array(100,100,0.4)
print(val_e1_p3_100)
collections.Counter(val_e1_p3_100)

graficar_histograma(val_e1_p3_10, 10, 0.4)
graficar_histograma(val_e1_p3_20, 20, 0.4)
graficar_histograma(val_e1_p3_50, 50, 0.4)
graficar_histograma(val_e1_p3_100, 100, 0.4)


# CONCLUSION: Teniendo en cuenta el p=0.4, puedo estimar que el pico de frencuencia en cada uno, va a estar determinado por el
# valor del rango que sea valor_maximo*0.4 (aproximadamente)