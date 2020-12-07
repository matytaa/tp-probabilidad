import scipy.stats as stats
from ejercicios.parte_1_simulacion.normal import fn_normal_array
from funciones.funciones import media_muestral
from funciones.funciones import varianza_muestral
from funciones.funciones import con_varianza_conocida
from funciones.funciones import con_varianza_desconocida
from funciones.funciones import verificar_hipotesis_alternativa

## Parte 4: Estadística inferencial
##  Parte 4 Ejercicio 1

# Generar dos muestras N(100, 5), una de tamaño n = 10 y otra de tamaño n = 30.
# Obtener estimaciones puntuales de su media y varianza.

val_e1_p4_10 = fn_normal_array(10,100, 5)
print(val_e1_p4_10)

media_muestral_n10 = media_muestral(val_e1_p4_10)
print("Media muestral n=10: ",media_muestral_n10)
print("Varianza muestral: ", varianza_muestral(val_e1_p4_10))

val_e1_p4_30 = fn_normal_array(30,100, 5)
print(val_e1_p4_30)

media_muestral_n30 = media_muestral(val_e1_p4_30)
print("Media muestral n=30: ", media_muestral_n30)

varianza_muestral_n30 = varianza_muestral(val_e1_p4_30)
print("Varianza muestral: ", varianza_muestral_n30)



# # Parte 4 Ejercicio 2
# Suponga que ya conoce el dato de que la distribución tiene varianza 5.
# Obtener intervalos de confianza del 95% y 98% para la media de ambas muestras.


mu = media_muestral_n10  # N(100,5) n=10
varianza = 5
n = 10  # Como es poblacion normal, puede usarse la formula
confianza = 0.95
limite_inferior, limite_superior = con_varianza_conocida(varianza, n, confianza, mu)
print("Intervalo de confianza del 95% se encuentra entre: " + str(limite_inferior) + " <= mu <= " + str(limite_superior))

confianza = 0.98
limite_inferior, limite_superior = con_varianza_conocida(varianza, n, confianza, mu)
print("Intervalo de confianza del 98% se encuentra entre: " + str(limite_inferior) + " <= mu <= " + str(limite_superior))

mu = media_muestral_n30  # N(100,5) n=30
varianza = 5
n = 30
confianza = 0.95
limite_inferior, limite_superior = con_varianza_conocida(varianza, n, confianza, mu)
print("Intervalo de confianza del 95% se encuentra entre: " + str(limite_inferior) + " <= mu <= " + str(limite_superior))

confianza = 0.98
limite_inferior, limite_superior = con_varianza_conocida(varianza, n, confianza, mu)
print("Intervalo de confianza del 98% se encuentra entre: " + str(limite_inferior) + " <= mu <= " + str(limite_superior))



# # Parte 4 Ejercicio 3
# Repita el punto anterior pero usando la varianza estimada s² , para la muestra de tamaño adecuado.
# Intervalo de confianza con varianza desconocida
N = 10
valores = val_e1_p4_10
mu = media_muestral_n10

print("Valor de mu:",mu)

confianza = 0.95
limite_inferior, limite_superior = con_varianza_desconocida(N,confianza, valores, mu)
print("Intervalo de confianza del 95% se encuentra entre: "+ str(limite_inferior) + " <= mu <= " + str(limite_superior))

confianza = 0.98
limite_inferior, limite_superior = con_varianza_desconocida(N,confianza, valores, mu)
print("Intervalo de confianza del 98% se encuentra entre: "+ str(limite_inferior) + " <= mu <= " + str(limite_superior))

N = 30
valores = val_e1_p4_30
mu = media_muestral_n30

print("Valor de mu:",mu)

confianza = 0.95
limite_inferior, limite_superior = con_varianza_desconocida(N, confianza, valores, mu)
print("Intervalo de confianza del 95% se encuentra entre: "+ str(limite_inferior) + " <= mu <= " + str(limite_superior))

confianza = 0.98
limite_inferior, limite_superior = con_varianza_desconocida(N, confianza, valores, mu)
print("Intervalo de confianza del 98% se encuentra entre: "+ str(limite_inferior) + " <= mu <= " + str(limite_superior))


# # Parte 4 Ejercicio 4
# Probar a nivel 0,99 la hipótesis de que la varianza sea σ² > 5. Calcular la probabilidad de cometer
# error tipo II para la hipótesis alternativa σ² = 6.

n = 10
s_cuadrado = varianza_muestral(val_e1_p4_10)
alfa = 1 - 0.99
grados_libertad = n - 1
varianza = 5

chi_cuadrado_tabla = 21.666 # se obtuvo por tabla para alfa = 0.01 y grados de libertad = 9
chi_cuadrado_calculado = (grados_libertad * s_cuadrado) / varianza

h_nula =        "tiene que ser igual que σ^2 = 5"
h_alternativa = "tiene que ser mayor que σ^2 > 5"


cumple = verificar_hipotesis_alternativa(chi_cuadrado_tabla, chi_cuadrado_calculado)

if (cumple):
    print("Para chi cuadrada " + str(chi_cuadrado_calculado) + ", hay evidencia para rechazar hipotesis nula")
else:
    print("Para chi cuadrada " + str(chi_cuadrado_calculado) + ", no hay evidencia para rechazar hipotesis nula")

n = 30
s_cuadrado = varianza_muestral(val_e1_p4_30)
alfa = 1 - 0.99
grados_libertad = n - 1
varianza = 5

chi_cuadrado_tabla = 49.588 # se obtuvo por tabla para alfa = 0.01 y grados de libertad = 29
chi_cuadrado_calculado = (grados_libertad * s_cuadrado) / varianza

h_nula =        "tiene que ser igual que σ^2 = 5"
h_alternativa = "tiene que ser mayor que σ^2 > 5"

cumple = verificar_hipotesis_alternativa(chi_cuadrado_tabla, chi_cuadrado_calculado)

if (cumple):
    print("Para chi cuadrada " + str(chi_cuadrado_calculado) + ", hay evidencia para rechazar hipotesis nula")
else:
    print("Para chi cuadrada " + str(chi_cuadrado_calculado) + ", no hay evidencia para rechazar hipotesis nula")

# Calcular la probabilidad de cometer error tipo II para la hipótesis alternativa σ² = 6. Para n = 10
n = 10
varianza_nueva = 6
grados_libertad = n - 1

chi_cuadrado_tabla = 21.666 # se obtuvo por tabla para alfa = 0.01 y grados de libertad = 9
s_cuadrado_l = (chi_cuadrado_tabla * varianza) / grados_libertad
chi_cuadrado_calculado = (grados_libertad * s_cuadrado_l) / varianza_nueva

beta = stats.chi2.cdf(chi_cuadrado_calculado, grados_libertad)
print("La probabilidad de cometer error tipo II para n = 10 es:", beta)

# Calcular la probabilidad de cometer error tipo II para la hipótesis alternativa σ² = 6. Para n = 30
n = 30
varianza_nueva = 6
grados_libertad = n - 1

chi_cuadrado_tabla = 49.588 # se obtuvo por tabla para alfa = 0.01 y grados de libertad = 29
s_cuadrado_l = (chi_cuadrado_tabla * varianza) / grados_libertad
chi_cuadrado_calculado = (grados_libertad * s_cuadrado_l) / varianza_nueva

beta = stats.chi2.cdf(chi_cuadrado_calculado, grados_libertad)
print("La probabilidad de cometer error tipo II para n = 30 es:", beta)