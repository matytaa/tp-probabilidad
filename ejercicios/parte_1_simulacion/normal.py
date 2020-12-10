import numpy as np
import math

# # Ejercicio 4
# Investigar como generar números aleatorios con distribución normal. Implementarlo.
# Usamos el método de Box-muller para generar numeros random siguiendo una distribucion normal
# https://es.wikipedia.org/wiki/M%C3%A9todo_de_Box-Muller
# Box-muller se calcula tanto con el seno como el coseno, pero es indistinto con cual te quedas
# en nuestro caso nos quedamos con el seno

def random_gaussiano(mean, desviacion_estandar):
    theta = 2 * math.pi * np.random.uniform(0,1)
    rho = math.sqrt(-2 * np.log(1 - np.random.uniform(0,1)))
    scale = desviacion_estandar * rho
    y = mean + scale * math.sin(theta)
    return y

def obtener_muestras_normales(casos, media, desviacion_estandar):
    valores = np.zeros(casos)
    for i in range(0,casos):
        valores[i-1] = random_gaussiano(media, desviacion_estandar)
    return valores
print("Inicio PRUEBA")
normal = obtener_muestras_normales(30, 100, 5)
print(normal)
print(np.mean(normal))
print("FIN PRUEBA")
