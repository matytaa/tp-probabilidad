import math
import scipy.stats as stats

def media_muestral(valores):
    suma = 0
    for i in range(0, len(valores)):
        suma += valores[i]
    return suma / len(valores)


def varianza_muestral(valores):
    media = media_muestral(valores)
    suma = 0
    for i in range(0, len(valores)):
        suma += math.pow(valores[i] - media, 2)

    varianza = suma / (len(valores) - 1)
    return varianza

def con_varianza_conocida(varianza, n, confianza, mu):
    sigma = math.sqrt(varianza)
    alfa = 1 - confianza
    alfa_sobre_2 = alfa / 2
    z_alfa_sobre_2 = stats.norm.ppf(alfa_sobre_2)  # Busco el valor de la normal en tabla
    x_raya = mu  # media muestral
    limite_inferior = x_raya + (z_alfa_sobre_2 * (sigma / math.sqrt(n)))
    limite_superior = x_raya - (z_alfa_sobre_2 * (sigma / math.sqrt(n)))
    return limite_inferior, limite_superior


def obtener_s(valores):
    s2 = varianza_muestral(valores)
    return math.sqrt(s2)


def calcular_n(zeta_alfa, sigma, e):
    return math.pow((zeta_alfa * sigma) / e, 2)


def con_varianza_desconocida(N, confianza, valores, mu):
    x_raya = mu  # media muestral
    alfa = 1 - confianza
    alfa_sobre_2 = (alfa / 2)
    s = obtener_s(valores)  # desvio estanda
    zeta_alfa = -round(stats.norm.ppf(alfa_sobre_2), 2)  # Busco el valor de la normal en tabl
    error_estandar = (s / math.sqrt(N))  # zeta_alfa*
    n_muestra = int(calcular_n(zeta_alfa, s, error_estandar))
    print("n Muestra:", n_muestra, " > ", N)
    t_alfa_sobre_2 = stats.t.ppf(alfa_sobre_2, n_muestra - 1)  # Busco el valor de la T-Student en tabla
    limite_inferior = x_raya + (t_alfa_sobre_2 * (s / math.sqrt(n_muestra)))
    limite_superior = x_raya - (t_alfa_sobre_2 * (s / math.sqrt(n_muestra)))

    return limite_inferior, limite_superior


def verificar_hipotesis_alternativa(chi_cuadrado_tabla, chi_cuadrado_calculado):
    return chi_cuadrado_calculado > chi_cuadrado_tabla
