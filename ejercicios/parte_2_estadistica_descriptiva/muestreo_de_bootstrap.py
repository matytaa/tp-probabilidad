import numpy as np

def sampleo_bootstrap(muestra, casos):
    tamanio_de_la_muestra = len(muestra)
    muestreo_de_la_media = []
    for _ in range(casos):
        muestreo_con_reemplazo = np.random.choice(muestra, size=tamanio_de_la_muestra)
        muestreo_de_la_media.append(muestreo_con_reemplazo.mean())
    return muestreo_de_la_media