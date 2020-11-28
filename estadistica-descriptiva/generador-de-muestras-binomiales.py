import collections
import matplotlib.pyplot as plt
from funciones.binomial import fn_binomial_array
from statsmodels.distributions.empirical_distribution import ECDF

# # Ejercicio 3
# Generar una muestra de números Bin(10, 0.3) de tamaño de muestra N = 50.
# Construir la función de distribución empírica de dicha muestra.

def función_de_distribución_acumulativa_empirica(casos, n, p):
    val_e3 = fn_binomial_array(casos, n, p)
    print(val_e3)
    collections.Counter(val_e3)
    ecdf = ECDF(val_e3)
    print('P(x<0): %.3f' % ecdf(0))
    print('P(x<1): %.3f' % ecdf(1))
    print('P(x<2): %.3f' % ecdf(2))
    print('P(x<3): %.3f' % ecdf(3))
    print('P(x<4): %.3f' % ecdf(4))
    print('P(x<5): %.3f' % ecdf(5))
    print('P(x<6): %.3f' % ecdf(6))

    plt.figure("Muestra de números Bin(10, 0.3) de tamaño de muestra N = 50")
    plt.plot(ecdf.x, ecdf.y)
    plt.show()


casos = 50
n = 10
p = 0.3
función_de_distribución_acumulativa_empirica(casos, n, p)