from random import choices
from funciones.funciones import media_muestral


def generar_muestra_segun_cdf(distribucion_empirica, casos):
    medias = []
    largo = len(distribucion_empirica)
    valores = []
    for i in range(0, largo):
        valores.append(distribucion_empirica[i][1])
    for i in range(0, casos):
        muestra_nueva = choices(valores, k=largo)
        media = media_muestral(muestra_nueva)
        medias.append(media)
    return medias