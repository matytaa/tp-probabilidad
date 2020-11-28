import numpy as np
import matplotlib.pyplot as plt
import collections
from scipy import stats


# La Distribución de Bernoulli describe un experimento probabilístico en donde el ensayo
# tiene dos posibles resultados, éxito o fracaso.

# p   es la probabilidad de éxito
# 1−p es la probabilidad de fracaso

# np.random.uniform(0,1) "la función que genera un número aleatorio uniforme entre 0 y 1"
# De esta forma, devuelve valores equiprobables entre 0 y 1
# Los valores, los voy a generar de esta forma valor = np.random.uniform(0,1) y dps, comparo con el valor de p (prob)

def fn_bernoulli_random(p):
        if np.random.uniform(0,1) > p:
            return 0
        else:
            return 1

# implemente una función que genere un array de valores distribuido Bernoulli con probabilidad p.
def fn_bernoulli_array(x,p):
    valores = np.zeros((x))
    for i in range(0,x):
        valores[i-1] = fn_bernoulli_random(p)
    return valores

p=100
x=4
val = fn_bernoulli_array(100,0.4)
print(val)
collections.Counter(val)


#bernoulli = stats.bernoulli(p)
## Función de Masa de Probabilidad
#fmp = bernoulli.pmf(x)

## Graficando Bernoulli
#fig, ax = plt.subplots()
#ax.plot(x, fmp, 'bo')
#ax.vlines(x, 0, fmp, colors='b', lw=5, alpha=0.5)
#ax.set_yticks([0., 0.2, 0.4, 0.6])

#plt.title('Distribución Bernoulli')
#plt.ylabel('probabilidad')
#plt.xlabel('valores')
#plt.show()