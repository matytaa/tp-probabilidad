#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Parte 1: Simulación

#En esta primera parte, construiremos varios generadores de números aleatorios que usaremos para obtener muestras con distribución conocida sobre las que vamos a trabajar posteriormente.

1. Utilizando únicamente la función random de su lenguaje (la función que genera un número aleatorio uniforme entre 0 y 1),
implemente una función que genere un número distribuido Bernoulli con probabilidad p.
2. Utilizando la función del punto anterior, implemente otra que genere un número binomial con los parámetros n,p.

3. Utilizando el procedimiento descrito en el capítulo 6 del Dekking (método de la función inversa o de Monte Carlo), imple-
mentar una función que permita generar un número aleatorio con distribución E xp(λ).

4. Investigar como generar números aleatorios con distribución normal. Implementarlo.

Parte 2: Estadística descriptiva
Ahora vamos a aplicar las técnicas vistas en la materia al estudio de algunas muestras de datos.
1. Generar tres muestras de números aleatorios Exp(0,5) de tamaño n = 10, n = 30 y n = 200. Para cada una, computar la media
get_ipython().set_next_input('y varianza muestral. ¿Qué observa');get_ipython().run_line_magic('pinfo', 'observa')
2. Para las tres muestras anteriores, graficar los histogramas de frecuencias relativas con anchos de banda 0,4, 0,2 y 0,1; es decir,
get_ipython().set_next_input('un total de 9 histogramas. ¿Qué conclusiones puede obtener');get_ipython().run_line_magic('pinfo', 'obtener')
3. Generar una muestra de números Bin(10, 0,3) de tamaño n = 50. Construir la función de distribución empírica de dicha
muestra.
4. A partir de la función de distribución empírica del punto anterior, generar una nueva muestra de números aleatorios utili-
zando el método de simulación de la primera parte. Computar la media y varianza muestral y graficar el histograma.
5. Repetir el experimento de los dos puntos anteriores con dos muestras aleatorias más generadas con los mismos parámetros.
get_ipython().set_next_input('¿Qué conclusión saca');get_ipython().run_line_magic('pinfo', 'saca')

Parte 3: Convergencia
El propósito de esta sección es ver en forma práctica los resultados de los teoremas de convergencia.
1. Generar cuatro muestras de números aleatorios de tamaño 100, todas con distribución binomial con p = 0,40 y n = 10, n = 20,
get_ipython().set_next_input('n = 50 y n = 100 respectivamente. Graficar sus histogramas. ¿Qué observa');get_ipython().run_line_magic('pinfo', 'observa')
2. Elija la muestra de tamaño 200 y calcule la media y desviación estándar muestral. Luego, normalice cada dato de la muestra
y grafique el histograma de la muestra normalizada. Justifique lo que observa.
3. Para cada una de las muestras anteriores, calcule la media muestral. Justifique lo que observa.

Parte 4: Estadística inferencial
Para terminar, vamos a hacer inferencia con las muestras que generamos y obtener así información sobre sus distribuciones.
1. Generar dos muestras N(100, 5), una de tamaño n = 10 y otra de tamaño n = 30. Obtener estimaciones puntuales de su media
y varianza.
2. Suponga que ya conoce el dato de que la distribución tiene varianza 5. Obtener intervalos de confianza del 95% y 98% para
la media de ambas muestras.
3. Repita el punto anterior pero usando la varianza estimada s^2, para la muestra de tamaño adecuado.
4. Probar a nivel 0,99 la hipótesis de que la varianza sea σ^2 > 5. Calcular la probabilidad de cometer error tipo II 
para la hipótesisalternativa σ^2 = 6.
5. Agrupando los datos en subgrupos de longitud 0,5, probar a nivel 0,99 la hipótesis de que la muestra proviene 
de una distribución normal.


# In[1]:


#Imports que voy a necesitar

import numpy as np # importando numpy
from scipy import stats # importando scipy.stats
import pandas as pd # importando pandas
import matplotlib.pyplot as plt # importando matplotlib
import seaborn as sns # importando seaborn
# importanto la api de statsmodels
import statsmodels.formula.api as smf
import statsmodels.api as sm
import math
import collections 
from statsmodels.distributions.empirical_distribution import ECDF


# In[3]:


# Bibliografia
# https://relopezbriega.github.io/blog/2016/06/29/distribuciones-de-probabilidad-con-python
# https://relopezbriega.github.io/blog/2015/06/27/probabilidad-y-estadistica-con-python/
# Fn Binomial: https://www.aglarick.com/2020/02/15/generacion-de-la-distribucion-binomial-en-python-con-jupyter-y-matplotlib/
# Fn Normal: https://stackoverrun.com/es/q/3329235


# # Parte 1: Simulación

# # Parte 1 Ejercicio 1

# In[4]:


# En esta primera parte, construiremos varios generadores de números aleatorios que usaremos para obtener muestras con distribu-
# ción conocida sobre las que vamos a trabajar posteriormente.

# 1. Utilizando únicamente la función random de su lenguaje (la función que genera un número aleatorio uniforme entre 0 y 1),
# implemente una función que genere un número distribuido Bernoulli con probabilidad p.


# In[3]:


#Fijo la semilla del random para que siempre sean los mismos datos 
np.random.seed(1)


# In[2]:


# np.random.uniform(0,1) "la función que genera un número aleatorio uniforme entre 0 y 1"
# De esta forma, devuelve valores equiprobables entre 0 y 1
# Los valores, los voy a generar de esta forma valor = np.random.uniform(0,1) y dps, comparo con el valor de p (prob)

def fn_bernoulli_random(p):
        if np.random.uniform(0,1) > p:
            return 0
        else:
            return 1


# In[3]:


# implemente una función que genere un array de valores distribuido Bernoulli con probabilidad p.
def fn_bernoulli_array(x,p):
    valores = np.zeros((x))
    for i in range(0,x):
        valores[i-1] = fn_bernoulli_random(p)
    return valores
        


# In[4]:


datos_bernoulli = fn_bernoulli_array(100,0.4)
print(datos_bernoulli)
print ("Número + Cantidad de ocurrencias encontradas ==> %s" % collections.Counter(datos_bernoulli))


# # Parte 1 Ejercicio 2

# 2. Utilizando la función del punto anterior, implemente otra que genere un número binomial con los parámetros n,p.

# In[5]:


def fn_binomial_random(n,p):
    intentos=[np.random.uniform(0,1) for x in range(0,n)]
    exitos=[intento<=p for intento in intentos]
    return sum(exitos)


# In[6]:


# Implemento funcion para retornar un conjunto de valores aleatorios siguiendo la distribucion binomial
def fn_binomial_array(casos,n,p):
    valores = np.zeros(casos)
    for i in range(0,casos):
        valores[i-1] = fn_binomial_random(n,p)
    return valores
        


# In[7]:


exitos = fn_binomial_random(50, 0.4)
print("%s éxitos de una Binomial con una muestra de %s y probabilidad %s" % (exitos, 50, 0.4))


# In[8]:


datos_binomial = fn_binomial_array(1000,6,0.5)
print(datos_binomial)
print ("Número + Cantidad de ocurrencias encontradas ==> %s" % collections.Counter(datos_binomial))


# In[9]:


def graficaBinomial(puntostotales, n, p):
    Xs=[k/n    for k in range(0,n+1)]
    Ys=[0    for i in range(0,n+1)]
    puntoactual=0
    while puntoactual<puntostotales:
        ubicacion=fn_binomial_random(n, p)
        Ys[ubicacion]+=1
        puntoactual+=1
    return Xs, Ys


# In[10]:


ns=[6]
p=0.5
puntos=1000
for n in ns:
    curva=graficaBinomial(puntos, n, p)
    plt.plot(*curva, label=f'n = {n}')
plt.xlabel('Probabilidad ')
plt.ylabel('')
plt.title('Histograma Binomial')
plt.legend()
plt.show()


# # Parte 1 Ejercicio 3

# In[15]:


# Utilizando el procedimiento descrito en el capítulo 6 del Dekking (método de la función inversa o de Monte Carlo), imple-
# mentar una función que permita generar un número aleatorio con distribución Exp(λ).


# Metodo de funcion inversa: pagina 74 del Dekkings

# ![image.png](attachment:image.png)

# In[11]:


def fn_inversa_exponencial(_lambda, u):
    return -(1/_lambda)* math.log10(u) 


# In[12]:


def fn_exponencial_random(_lambda):
    numeroRandomConDistribucionUniforme = np.random.uniform(0,1)
    result = fn_inversa_exponencial(_lambda, numeroRandomConDistribucionUniforme)
    return result


# # Parte 1 Ejercicio 4

# In[19]:


# Investigar como generar números aleatorios con distribución normal. Implementarlo.


# Usamos el método de Box-muller para generar numeros random siguiendo una distribucion normal 
# https://es.wikipedia.org/wiki/M%C3%A9todo_de_Box-Muller

# In[43]:


def fn_gaussian_random(mean, stddev): 
    theta = 2 * math.pi * np.random.uniform(0,1) 
    rho = math.sqrt(-2 * math.log10(1 - np.random.uniform(0,1))) 
    scale = stddev * rho 
    x = mean + scale * math.cos(theta) 
    y = mean + scale * math.sin(theta) 
    return y


# In[44]:


def fn_normal_array(casos,mean,stddev):
    valores = np.zeros(casos)
    for i in range(0,casos):
        valores[i-1] = fn_gaussian_random(mean, stddev)
    return valores


# In[46]:


datos_normal = fn_normal_array(20,6.5, 0.5)
print(datos_normal)
print(np.mean(datos_normal))


# # Parte 2: Estadística descriptiva

# In[23]:


# Ahora vamos a aplicar las técnicas vistas en la materia al estudio de algunas muestras de datos.


# # Parte 2 Ejercicio 1

# In[24]:


# Generar tres muestras de números aleatorios Exp(0,5) de tamaño n = 10, n = 30 y n = 200. Para cada una, computar la media
# y varianza muestral. ¿Qué observa?


# In[16]:


def media_muestral(valores):
    suma = 0
    for i in range(0, len(valores)):
        suma += valores[i]
    
    return suma/len(valores)

def varianza_muestral(valores):
    #suma = 0
    #suma_parcial = 0
    #for i in range(0, len(valores)):
    #    suma += valores[i]
    #for i in range(0, len(valores)):
    #    suma_parcial += (valores[i]- suma)**2
    
    #return suma_parcial/len(valores)
    suma = 0
    for i in range(0,len(valores)):
        suma = suma + math.pow(valores[i], 2)

    varianza = suma/10 - math.pow(media_muestral(valores), 2)
    return varianza


# In[17]:


def fn_generador_de_muestras_numeros_random_con_dist_exponencial(n,_lambda):
    valores = np.zeros(n)
    for i in range(0,n):
        valores[i-1] = fn_exponencial_random(_lambda)
    
    #Calculo de media_muestral
    rt_media_muestral = media_muestral(valores)
    
    #Calculo de varianza_muestral
    rt_varianza_muestral = varianza_muestral(valores)

    return valores, rt_media_muestral, rt_varianza_muestral


# Primera muestra con n=10: 

# In[18]:


val_n10, media, varianza = fn_generador_de_muestras_numeros_random_con_dist_exponencial(10,0.5)
print(val_n10)
#collections.Counter(val_n10) 
print("Media=", media)
print("Varianza=", varianza)


# Segunda muestra con n=30: 

# In[19]:


val_n30, media, varianza = fn_generador_de_muestras_numeros_random_con_dist_exponencial(30,0.5)
print(val_n30)
#collections.Counter(val_n30) 
print("Media=", media)
print("Varianza=", varianza)


# Segunda muestra con n=200:

# In[20]:


val_n200, media, varianza = fn_generador_de_muestras_numeros_random_con_dist_exponencial(200,0.5)
print(val_n200)
#collections.Counter(val_n200) 
print("Media=", media)
print("Varianza=", varianza)


# # Parte 2 Ejercicio 2

# In[30]:


# Para las tres muestras anteriores, graficar los histogramas de frecuencias relativas con anchos de banda 0,4, 0,2 y 0,1; es decir,
# un total de 9 histogramas. ¿Qué conclusiones puede obtener?


# In[21]:


inicio = int(min(val_n10))
fin = int(max(val_n10))
print("Inicio=", inicio)
print("Fin=", fin)
ancho = 0.4
div = np.linspace(inicio,fin,round(1+(fin-inicio)/ancho))
plt.hist(val_n10,div)
plt.title('Histograma Ancho 0.4')
plt.grid()
plt.show()
ancho = 0.2
div = np.linspace(inicio,fin,round(1+(fin-inicio)/ancho))
plt.hist(val_n10,div)
plt.title('Histograma Ancho 0.2')
plt.grid()
plt.show()
ancho = 0.1
div = np.linspace(inicio,fin,round(1+(fin-inicio)/ancho))
plt.hist(val_n10,div)
plt.title('Histograma Ancho 0.1')
plt.grid()
plt.show()


# In[22]:


inicio = int(min(val_n30))
fin = int(max(val_n30))
ancho = 0.4
print("Inicio=", inicio)
print("Fin=", fin)
div = np.linspace(inicio,fin,round(1+(fin-inicio)/ancho))
plt.hist(val_n30,div)
plt.title('Histograma Ancho 0.4')
plt.grid()
plt.show()
ancho = 0.2
div = np.linspace(inicio,fin,round(1+(fin-inicio)/ancho))
plt.hist(val_n30,div)
plt.title('Histograma Ancho 0.2')
plt.grid()
plt.show()
ancho = 0.1
div = np.linspace(inicio,fin,round(1+(fin-inicio)/ancho))
plt.hist(val_n30,div)
plt.title('Histograma Ancho 0.1')
plt.grid()
plt.show()


# In[23]:


inicio = int(min(val_n200))
fin = int(max(val_n200))
ancho = 0.4
print("Inicio=", inicio)
print("Fin=", fin)
div = np.linspace(inicio,fin,round(1+(fin-inicio)/ancho))
plt.hist(val_n200,div)
plt.title('Histograma Ancho 0.4')
plt.grid()
plt.show()
ancho = 0.2
div = np.linspace(inicio,fin,round(1+(fin-inicio)/ancho))
plt.hist(val_n200,div)
plt.title('Histograma Ancho 0.2')
plt.grid()
plt.show()
ancho = 0.1
div = np.linspace(inicio,fin,round(1+(fin-inicio)/ancho))
plt.hist(val_n200,div)
plt.title('Histograma Ancho 0.1')
plt.grid()
plt.show()


# # Parte 2 Ejercicio 3

# In[38]:


# Generar una muestra de números Bin(10, 0.3) de tamaño de muestra N = 50. Construir la función de distribución empírica de dicha
# muestra.


# In[ ]:


# Referencia Funcion de Distribucion Empirica
# https://machinelearningmastery.com/empirical-distribution-function-in-python/#:~:text=An%20empirical%20distribution%20function%20can,specific%20observations%20from%20the%20domain.


# In[24]:


def funcion_de_distribución_acumulativa_empirica(array_de_valores, limite_superior):
    mapa_de_valores = collections.Counter(array_de_valores)
    keys = sorted(mapa_de_valores.keys())
    suma = 0
    for i in range(0, len(keys)):
       if(keys[i] <= limite_superior):
           suma += mapa_de_valores[keys[i]]
    return suma/ len(array_de_valores)


def funcion_de_distribución_empirica(array_de_valores):
    mapa_de_valores = collections.Counter(array_de_valores)
    keys = sorted(mapa_de_valores.keys())
    lista_de_valores_de_acumuladas = np.zeros(len(keys))
    for i in range(0, len(keys)):
          lista_de_valores_de_acumuladas[i]= funcion_de_distribución_acumulativa_empirica(array_de_valores, keys[i])
    return dict(zip(keys, lista_de_valores_de_acumuladas))


# In[28]:


casos = 50
n = 10
p = 0.3

array_de_valores = fn_binomial_array(casos, n, p)
distribucion_empirica_ejercicio_3 = funcion_de_distribución_empirica(array_de_valores)

print("array de valores con distribucion binomial: " + str(array_de_valores))
print("array de valores de F empirica: ",distribucion_empirica_ejercicio_3)


# In[29]:


lists = sorted(distribucion_empirica_ejercicio_3.items()) 
x, y = zip(*lists)
plt.plot(x, y)
plt.title('Probabilidad Acumulada F Empirica')
plt.grid()
plt.show()


# In[30]:


inicio = int(min(array_de_valores)) -1 
fin = int(max(array_de_valores)) + 1

ancho = 0.1
print("Inicio=", inicio)
print("Fin=", fin)
div = np.linspace(inicio,fin,round(1+(fin-inicio)/ancho))
plt.hist(array_de_valores,div)
plt.title('Histograma Muestra 1')
plt.grid()
plt.show()


# In[154]:


ecdf = ECDF(array_de_valores)
print('P(x<0): %.3f' % ecdf(0))
print('P(x<1): %.3f' % ecdf(1))
print('P(x<2): %.3f' % ecdf(2))
print('P(x<3): %.3f' % ecdf(3))
print('P(x<4): %.3f' % ecdf(4))
print('P(x<5): %.3f' % ecdf(5))
print('P(x<6): %.3f' % ecdf(6))
# plot the cdf
plt.plot(ecdf.x, ecdf.y)
plt.show()


# In[55]:


datos_normal = fn_normal_array(20,6.5, 0.5)
datos_normal = datos_normal.round(2)

print(datos_normal)
print(np.mean(datos_normal))

ecdf = ECDF(datos_normal)

# plot the cdf
plt.plot(ecdf.x, ecdf.y)
plt.show()

inicio = int(min(datos_normal)) -1 
fin = int(max(datos_normal)) + 1

ancho = 0.1
print("Inicio=", inicio)
print("Fin=", fin)
div = np.linspace(inicio,fin,round(1+(fin-inicio)/ancho))
plt.hist(datos_normal,div)
plt.title('Histograma Muestra 1')
plt.grid()
plt.show()


# # Parte 2 Ejercicio 4

# In[ ]:


# A partir de la función de distribución empírica del punto anterior, generar una nueva muestra de números aleatorios utilizando
# el método de simulación de la primera parte. 
# Computar la media y varianza muestral y graficar el histograma.


# Ejemplos de boostrap:
# https://datasciencechalktalk.com/2019/11/12/bootstrap-sampling-an-implementation-with-python/
# https://www.linkedin.com/learning/r-para-data-scientist-avanzado/bootstrap-en-r-muestreo?originalSubdomain=es

# In[42]:


def sampleo_bootstrap(array_de_valores,casos):
    len_array = len(array_de_valores)
    sample = np.random.choice(array_de_valores, size=len_array) #so n=300

    sample_mean = []
    for _ in range(casos):  #so B=10000
        sample_n = np.random.choice(sample, size=len_array)
    #    print(sample_n)
        sample_mean.append(sample_n.mean())

    return sample_mean    

sample = sampleo_bootstrap(array_de_valores,1000)

#print(sample)

plt.hist(sample)
plt.grid()
plt.show()


# In[40]:


casos = 50
n = 10
p = 0.3
array_de_valores_ejercicio_4 = fn_binomial_array(casos, n, p)
distribucion_empirica_ejercicio_4 = funcion_de_distribución_empirica(array_de_valores_ejercicio_4)

print(distribucion_empirica_ejercicio_4)
print ("Número + Cantidad de ocurrencias encontradas ==> %s" % collections.Counter(array_de_valores_ejercicio_4))


# In[41]:


inicio = int(min(array_de_valores_ejercicio_4)) -1 
fin = int(max(array_de_valores_ejercicio_4)) + 1

ancho = 0.1
print("Inicio=", inicio)
print("Fin=", fin)
div = np.linspace(inicio,fin,round(1+(fin-inicio)/ancho))
plt.hist(array_de_valores_ejercicio_4,div)
plt.title('Histograma Muestra 1')
plt.grid()
plt.show()


# In[ ]:


medias_muestrales = np.zeros(2)

lista_keys_de_distribucion_empirica_ejercicio_3 = []
for key,val in distribucion_empirica_ejercicio_3.items():
    lista_keys_de_distribucion_empirica_ejercicio_3.append(key)

lista_keys_de_distribucion_empirica_ejercicio_4 = []
for key,val in distribucion_empirica_ejercicio_4.items():
    lista_keys_de_distribucion_empirica_ejercicio_4.append(key)
    
medias_muestrales[0] = media_muestral(lista_keys_de_distribucion_empirica_ejercicio_3)
medias_muestrales[1] = media_muestral(lista_keys_de_distribucion_empirica_ejercicio_4)

print("varianza muestral funcion empirica array de valores ejericio 3 = ",
      varianza_muestral(lista_keys_de_distribucion_empirica_ejercicio_3))
print("varianza muestral funcion empirica array de valores ejericio 4 = ",
      varianza_muestral(lista_keys_de_distribucion_empirica_ejercicio_4))

print("Medias muestrales =",medias_muestrales)


# # Parte 2 Ejercicio 5

# In[66]:


# Repetir el experimento de los dos puntos anteriores con dos muestras aleatorias más generadas con los mismos parámetros.
# ¿Qué conclusión saca?


# In[163]:


casos = 50
n = 10
p = 0.3
array_de_valores_ejercicio_5_1 = fn_binomial_array(casos, n, p)
distribucion_empirica_ejercicio_5_1 = funcion_de_distribución_empirica(array_de_valores_ejercicio_5_1)

inicio = int(min(array_de_valores_ejercicio_5_1)) -1 
fin = int(max(array_de_valores_ejercicio_5_1)) + 1

ancho = 0.1
print("Inicio=", inicio)
print("Fin=", fin)
div = np.linspace(inicio,fin,round(1+(fin-inicio)/ancho))
plt.hist(array_de_valores_ejercicio_5_1,div)
plt.title('Histograma Muestra 1')
plt.grid()
plt.show()

array_de_valores_ejercicio_5_2 = fn_binomial_array(casos, n, p)
distribucion_empirica_ejercicio_5_2 = funcion_de_distribución_empirica(array_de_valores_ejercicio_5_2)

inicio = int(min(array_de_valores_ejercicio_5_2)) -1 
fin = int(max(array_de_valores_ejercicio_5_2)) + 1

ancho = 0.1
print("Inicio=", inicio)
print("Fin=", fin)
div = np.linspace(inicio,fin,round(1+(fin-inicio)/ancho))
plt.hist(array_de_valores_ejercicio_5_2,div)
plt.title('Histograma Muestra 2')
plt.grid()
plt.show()

medias_muestrales = np.zeros(4)

lista_keys_de_distribucion_empirica_ejercicio_5_1 = []
for key,val in distribucion_empirica_ejercicio_5_1.items():
    lista_keys_de_distribucion_empirica_ejercicio_5_1.append(key)

lista_keys_de_distribucion_empirica_ejercicio_5_2 = []
for key,val in distribucion_empirica_ejercicio_5_2.items():
    lista_keys_de_distribucion_empirica_ejercicio_5_2.append(key)
    
medias_muestrales[0] = media_muestral(lista_keys_de_distribucion_empirica_ejercicio_3)
medias_muestrales[1] = media_muestral(lista_keys_de_distribucion_empirica_ejercicio_4)
medias_muestrales[2] = media_muestral(lista_keys_de_distribucion_empirica_ejercicio_5_1)
medias_muestrales[3] = media_muestral(lista_keys_de_distribucion_empirica_ejercicio_5_2)


print("varianza muestral funcion empirica array de valores ejericio 3 = ",
      varianza_muestral(lista_keys_de_distribucion_empirica_ejercicio_3))
print("varianza muestral funcion empirica array de valores ejericio 4 = ",
      varianza_muestral(lista_keys_de_distribucion_empirica_ejercicio_4))
print("varianza muestral funcion empirica array de valores ejericio 5_1 = ",
      varianza_muestral(lista_keys_de_distribucion_empirica_ejercicio_5_1))
print("varianza muestral funcion empirica array de valores ejericio 5_2 = ",
      varianza_muestral(lista_keys_de_distribucion_empirica_ejercicio_5_2))


print("Medias Muestrales = ", medias_muestrales)

inicio = int(min(medias_muestrales)) -1 
fin = int(max(medias_muestrales)) + 1

ancho = 0.1
print("Inicio=", inicio)
print("Fin=", fin)
div = np.linspace(inicio,fin,round(1+(fin-inicio)/ancho))
plt.hist(medias_muestrales,div)
plt.title('Histograma de Medias Muestrales')
plt.grid()
plt.show()


# # Parte 3: Convergencia

# # Parte 3 Ejercicio 1

# In[ ]:


# Generar cuatro muestras de números aleatorios de tamaño 100, todas con distribución binomial con p = 0,40 y n = 10, n = 20,
# n = 50 y n = 100 respectivamente. Graficar sus histogramas. ¿Qué observa?


# In[68]:


val_e1_p3_10 = fn_binomial_array(100,10,0.4)
print(val_e1_p3_10)
#collections.Counter(val_e1_p3_10)


# In[69]:


val_e1_p3_20 = fn_binomial_array(100,20,0.4)
print(val_e1_p3_20)
#collections.Counter(val_e1_p3_20)


# In[70]:


val_e1_p3_50 = fn_binomial_array(100,50,0.4)
print(val_e1_p3_50)
#collections.Counter(val_e1_p3_50)


# In[71]:


val_e1_p3_100 = fn_binomial_array(100,100,0.4)
print(val_e1_p3_100)
#collections.Counter(val_e1_p3_100)


# In[110]:


inicio = int(min(val_e1_p3_10))
fin = int(max(val_e1_p3_10))
print("Inicio=", inicio)
print("Fin=", fin)
print("Valor pico de frecuencia=", 10*0.4)
ancho = 0.4
div = np.linspace(inicio,fin,round(1+(fin-inicio)/ancho))
plt.hist(val_e1_p3_10,div)
plt.title('Histograma Binomial n=10')
plt.grid()
plt.show()


# In[102]:


inicio = int(min(val_e1_p3_20))
fin = int(max(val_e1_p3_20))
print("Inicio=", inicio)
print("Fin=", fin)
print("Valor pico de frecuencia=", 20*0.4)
ancho = 0.4
div = np.linspace(inicio,fin,round(1+(fin-inicio)/ancho))
plt.hist(val_e1_p3_20,div)
plt.title('Binomial n=20')
plt.grid()
plt.show()


# In[108]:


inicio = int(min(val_e1_p3_50))
fin = int(max(val_e1_p3_50))
print("Inicio=", inicio)
print("Fin=", fin)
print("Valor pico de frecuencia=", 50*0.4)
ancho = 0.4
div = np.linspace(inicio,fin,round(1+(fin-inicio)/ancho))
plt.hist(val_e1_p3_50,div)
plt.title('Histograma Binomial n=50')
plt.grid()
plt.show()


# In[109]:


inicio = int(min(val_e1_p3_100))
fin = int(max(val_e1_p3_100))
print("Inicio=", inicio)
print("Fin=", fin)
print("Valor pico de frecuencia=", 100*0.4)
ancho = 0.4
div = np.linspace(inicio,fin,round(1+(fin-inicio)/ancho))
plt.hist(val_e1_p3_100,div)
plt.title('Histograma Binomial n=100')
plt.grid()
plt.show()


# In[76]:


# CONCLUSION: Teniendo en cuenta el p=0.4, puedo estimar que el pico de frencuencia en cada uno, va a estar determinado por el 
# valor del rango que sea valor_maximo*0.4 (aproximadamente)


# #  Parte 3 Ejercicio 2

# In[77]:


# Elija la muestra de tamaño 200 y calcule la media y desviación estándar muestral. Luego, normalice cada dato de la muestra
# y grafique el histograma de la muestra normalizada. Justifique lo que observa.


# In[78]:


val_e2_p3_200 = fn_binomial_array(200,10,0.4)
print(val_e2_p3_200)
#collections.Counter(val_e2_p3_200)


# In[79]:


# esperanza = n*p (binomial)
esperanza = 10 * 0.4
print("esperanza =",esperanza)

media_muestral_p3_2 = media_muestral(val_e2_p3_200)
print("media muestral = ",media_muestral_p3_2)


# In[80]:


# varianza = n*p*(1-p)
varianza = 10*0.4*(1-0.4)
print("varianza = ",varianza)

print("varianza muestral = ",
      varianza_muestral(val_e2_p3_200))


# In[105]:


inicio = int(min(val_e2_p3_200))
fin = int(max(val_e2_p3_200))
print("Inicio=", inicio)
print("Fin=", fin)
print("Valor pico de frecuencia=", media_muestral_p3_2)
ancho = 0.4
div = np.linspace(inicio,fin,round(1+(fin-inicio)/ancho))
plt.hist(val_e2_p3_200,div)
plt.title('Binomial n=200')
plt.grid()
plt.show()


# In[82]:


# Para normalizar el valor, tomo la siguiente normalización = ( x – min(x) ) / ( max(x) – min(x) )

minimo = (min(val_e2_p3_200))
maximo = (max(val_e2_p3_200))

calculo_b = maximo - minimo

normal = np.zeros((200))

for i in range(0,200):
    valor = val_e2_p3_200[i-1]
    calculo_a = valor - minimo
    calculo = calculo_a / calculo_b
    normal[i-1] = calculo

print(normal)


# In[83]:


df = pd.DataFrame({"valor":val_e2_p3_200,"normalizado":normal})


# In[84]:


print(df)


# In[107]:


inicio = int(min(normal))
fin = int(max(normal))
print("Inicio=", inicio)
print("Fin=", fin)

ancho = 0.05
div = np.linspace(inicio,fin,round(1+(fin-inicio)/ancho))
plt.hist(normal,div)
plt.title('Histograma Normal [0,1]')
plt.grid()
plt.show()


# In[85]:


# En el primer conjunto, la esperanza de la binomial, me da que en el valor 4 tendre el pico de frencuencia de ocurrencia
# Cuando normalizo los valores a un intervalo {0,1}, el valor representado de 4 es 0,5 esto en el histograma
# me muestra el valor con mayor frecuencia de ocurrencia.


# #  Parte 3 Ejercicio 3

# In[87]:


# Para cada una de las muestras anteriores, calcule la media muestral. Justifique lo que observa.


# In[87]:


print("media muestral ejercicio 1 n=10 = ",media_muestral(val_e1_p3_10))
print("media muestral ejercicio 1 n=20 = ",media_muestral(val_e1_p3_20))
print("media muestral ejercicio 1 n=50= ",media_muestral(val_e1_p3_50))
print("media muestral ejercicio 1 n=100= ",media_muestral(val_e1_p3_100))


print("media muestral ejercicio 2 n=10= ",media_muestral(val_e2_p3_200))


# # Parte 4: Estadistica inferencial

# #  Parte 4 Ejercicio 1

# In[88]:


# Generar dos muestras N(100, 5), una de tamaño n = 10 y otra de tamaño n = 30. Obtener estimaciones puntuales de su media
# y varianza.


# In[89]:


val_e1_p4_10 = fn_normal_array(10,100, 5)
print(val_e1_p4_10)


# In[96]:


val_e1_p4_30 = fn_normal_array(30,100, 5)
print(val_e1_p4_30)


# In[178]:


print("media muestral ejercicio 1 n=10 = ",media_muestral(val_e1_p4_10))


# In[179]:


print("varianza muestral = ",
      varianza_muestral(val_e1_p4_10))


# In[180]:


print("media muestral ejercicio 1 n=30 = ",media_muestral(val_e1_p4_30))


# In[181]:


print("varianza muestral = ",
      varianza_muestral(val_e1_p4_30))


# # Parte 4 Ejercicio 2

# In[34]:


# Suponga que ya conoce el dato de que la distribución tiene varianza 5.
# Obtener intervalos de confianza del 95% y 98% para la media de ambas muestras.


# In[46]:


def con_varianza_conocida(varianza, n, confianza, mu):
    sigma = math.sqrt(varianza)
    alfa = 1 - confianza
    alfa_sobre_2 = alfa / 2
    z_alfa_sobre_2 = stats.norm.ppf(alfa_sobre_2)
    x_raya = mu
    limite_inferior = x_raya + (z_alfa_sobre_2*(sigma/math.sqrt(n)))
    limite_superior = x_raya - (z_alfa_sobre_2*(sigma/math.sqrt(n)))
    return limite_inferior, limite_superior


# In[47]:


mu = 100
varianza = 5
n = 10

confianza = 0.95
limite_inferior, limite_superior = con_varianza_conocida(varianza, n, confianza, mu)
print("Intervalo de confianza del 95% se encuentra entre: " + str(limite_inferior) + " <= mu <= " + str(limite_superior))

confianza = 0.98
limite_inferior, limite_superior = con_varianza_conocida(varianza, n, confianza, mu)
print("Intervalo de confianza del 98% se encuentra entre: " + str(limite_inferior) + " <= mu <= " + str(limite_superior))


# # Parte 4 Ejercicio 3

# In[37]:


# Repita el punto anterior pero usando la varianza estimada s² , para la muestra de tamaño adecuado.


# In[48]:


def obtener_s(valores, x_raya, n):
    sumatoria = 0
    for i in range (0, len(valores)):
        sumatoria += math.pow(valores[i]-x_raya,2)
    return sumatoria/(n-1)

def con_varianza_desconocida(n, confianza, valores, mu):
    x_raya = mu
    alfa = 1 - confianza
    alfa_sobre_2 = alfa / 2
    s = obtener_s(valores, x_raya, n)
    t_alfa_sobre_2 = stats.t.ppf(alfa_sobre_2, n-1)
    limite_inferior = x_raya + (t_alfa_sobre_2*(s/math.sqrt(n)))
    limite_superior = x_raya - (t_alfa_sobre_2*(s/math.sqrt(n)))
    return limite_inferior, limite_superior


# In[49]:


confianza = 0.95
n = 10
valores = fn_normal_array(10,100, 5)
mu = 100
limite_inferior, limite_superior = con_varianza_desconocida(n, confianza, valores, mu)
print("Intervalo de confianza del 95% se encuentra entre: "+ str(limite_inferior) + " <= mu <= " + str(limite_superior))

confianza = 0.98
limite_inferior, limite_superior = con_varianza_desconocida(n, confianza, valores, mu)
print("Intervalo de confianza del 98% se encuentra entre: "+ str(limite_inferior) + " <= mu <= " + str(limite_superior)) 


# # Parte 4 Ejercicio 4

# In[41]:


# Probar a nivel 0,99 la hipótesis de que la varianza sea σ² > 5.
# Calcular la probabilidad de cometer error tipo II para la hipótesis alternativa σ² = 6.


# In[42]:


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


# In[50]:


alfa = 0.99
h_cero = "tiene que ser mayor que σ^2 5"
h_uno = "tiene que ser menor-igual que σ^2 5"
mu = 5
sigma_cuadrado = 5
n = 10
cumple = verificar_hipotesis_2(mu,n,sigma_cuadrado,alfa, ">")
print("La hipotesis h0:" + h_cero + " ¿Se cumple?:" + str(cumple)) 


# In[ ]:




