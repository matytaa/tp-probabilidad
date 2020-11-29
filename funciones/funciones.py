def esperanza_muestral(muestra):
    sumatoria = suma_de_valores(muestra)
    return sumatoria / len(muestra)

def varianza_muestral(muestra):
    varianza = 0
    sumatoria = suma_de_valores(muestra)
    largo_de_la_muestra = len(muestra)
    for i in range(0, largo_de_la_muestra):
        varianza = varianza + (muestra[i - 1] - sumatoria) ** 2
    return varianza / largo_de_la_muestra

def suma_de_valores(muestra):
    suma_b = 0
    for i in range(0, len(muestra)):
        suma_b = suma_b + muestra[i - 1]
    return suma_b