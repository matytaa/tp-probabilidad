import math
import scipy.stats as stats
from ejercicios.parte_1_simulacion.normal import fn_normal_array

# Probar a nivel 0,99 la hipótesis de que la varianza sea σ^2 > 5.
# Calcular la probabilidad de cometer error tipo II para la hipótesis alternativa σ^2 = 6.

def verificar_hipotesis(x_raya, mu, n, sigma_cuadrado, alfa, condicion):
    z_alfa = stats.norm.ppf(alfa)
    sigma = math.sqrt(sigma_cuadrado)
    zona_de_rechazo = (x_raya - mu) / (sigma / math.sqrt(n))

    if(condicion == "="):
        return z_alfa == zona_de_rechazo
    if(condicion == "<"):
        return zona_de_rechazo < z_alfa
    if(condicion == ">"):
        return z_alfa < zona_de_rechazo
    return z_alfa < zona_de_rechazo

def verificar_hipotesis_2(mu, n, sigma_cuadrado, alfa, condicion):
    z_alfa = stats.norm.ppf(alfa)
    sigma = math.sqrt(sigma_cuadrado)
    x_raya = mu+(z_alfa)*(sigma / math.sqrt(n))
    zona_de_rechazo = (x_raya - mu) / (sigma / math.sqrt(n))

    if(condicion == "="):
        return z_alfa == zona_de_rechazo
    if(condicion == "<"):
        return zona_de_rechazo < z_alfa
    if(condicion == ">"):
        return z_alfa < zona_de_rechazo
    return z_alfa < zona_de_rechazo

alfa = 0.99
h_cero = "tiene que ser mayor que σ^2 5"
h_uno = "tiene que ser menor-igual que σ^2 5"
mu = 5
sigma_cuadrado = 5
n = 10
cumple = verificar_hipotesis_2(mu,n,sigma_cuadrado,alfa, ">")
print("La hipotesis h0:" + h_cero + " ¿Se cumple?:" + str(cumple))