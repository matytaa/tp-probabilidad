import collections
from ejercicios.parte_1_simulacion.binomial import fn_binomial_array
# # Ejercicio 3
# Generar una muestra de números Bin(10, 0.3) de tamaño de muestra N = 50.
# Construir la función de distribución empírica de dicha muestra.

def función_de_distribución_acumulativa_empirica(array_de_valores, limite_superior):
    mapa_de_valores = collections.Counter(array_de_valores)
    keys = sorted(mapa_de_valores.keys())
    suma = 0
    for i in range(0, len(keys)):
       if(keys[i] <= limite_superior):
           suma += mapa_de_valores[keys[i]]
    return suma/ len(array_de_valores)

valor_limite = 5
casos = 50
n = 10
p = 0.3
valores_primera_tirada = fn_binomial_array(casos, n, p)
resultado = función_de_distribución_acumulativa_empirica(valores_primera_tirada, valor_limite)
print("bla: " + str(valores_primera_tirada))
print(resultado)


## Ejercicio 4
valores_segunda_tirada = fn_binomial_array(casos, n, p)
resultado2 = función_de_distribución_acumulativa_empirica(valores_segunda_tirada, valor_limite)
#print(resultado2)

## Ejercicio 5
valores_tercera_tirada = fn_binomial_array(casos, n, p)
resultado3 = función_de_distribución_acumulativa_empirica(valores_segunda_tirada, valor_limite)
#print(resultado3)

valores_tercera_tirada = fn_binomial_array(casos, n, p)
resultado4 = función_de_distribución_acumulativa_empirica(valores_segunda_tirada, valor_limite)
#print(resultado4)