from funciones.binomial import fn_binomial_array
from funciones.binomial import esperanza
from funciones.binomial import varianza
from funciones.normalizacion import normalizar
from histogramas.histograma import graficar_histograma
import collections

## Ejercicio 2
# Elija la muestra de tamaño 200 y calcule la media y desviación estándar muestral. Luego, normalice cada dato de la muestra
# y grafique el histograma de la muestra normalizada. Justifique lo que observa.


val_e2_p3_200 = fn_binomial_array(200,10,0.4)
print(val_e2_p3_200)
collections.Counter(val_e2_p3_200)

esperanza = esperanza(10, 0.4)
print(esperanza)

varianza = varianza(10,0.4)
print(varianza)

graficar_histograma(val_e2_p3_200, 200, 0.4)

normal = normalizar(val_e2_p3_200)
graficar_histograma(normal, 200, 0.05)

# En el primer conjunto, la esperanza de la binomial, me da que en el valor 4 tendre el pico de frencuencia de ocurrencia
# Cuando normalizo los valores a un intervalo {0,1}, el valor representado de 4 es 0,5 esto en el histograma
# me muestra el valor con mayor frecuencia de ocurrencia.