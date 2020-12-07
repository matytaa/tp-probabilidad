#!/usr/bin/env python
# coding: utf-8

# # TP Probabilidad y Estadística

# Integrantes: Victoria Vasquez - Leonel Campos - Gabriel Castelo

# Parte 1: Simulación
# 
# En esta primera parte, construiremos varios generadores de números aleatorios que usaremos para obtener muestras con distribu-
# ción conocida sobre las que vamos a trabajar posteriormente.
# 
# 1. Utilizando únicamente la función random de su lenguaje (la función que genera un número aleatorio uniforme entre 0 y 1),
# implemente una función que genere un número distribuido Bernoulli con probabilidad p.
# 2. Utilizando la función del punto anterior, implemente otra que genere un número binomial con los parámetros n,p.
# 
# 3. Utilizando el procedimiento descrito en el capítulo 6 del Dekking (método de la función inversa o de Monte Carlo), imple-
# mentar una función que permita generar un número aleatorio con distribución E xp(λ).
# 
# 4. Investigar como generar números aleatorios con distribución normal. Implementarlo.
# 
# Parte 2: Estadística descriptiva
# Ahora vamos a aplicar las técnicas vistas en la materia al estudio de algunas muestras de datos.
# 1. Generar tres muestras de números aleatorios Exp(0,5) de tamaño n = 10, n = 30 y n = 200. Para cada una, computar la media
# y varianza muestral. ¿Qué observa?
# 2. Para las tres muestras anteriores, graficar los histogramas de frecuencias relativas con anchos de banda 0,4, 0,2 y 0,1; es decir,
# un total de 9 histogramas. ¿Qué conclusiones puede obtener?
# 3. Generar una muestra de números Bin(10, 0,3) de tamaño n = 50. Construir la función de distribución empírica de dicha
# muestra.
# 4. A partir de la función de distribución empírica del punto anterior, generar una nueva muestra de números aleatorios utili-
# zando el método de simulación de la primera parte. Computar la media y varianza muestral y graficar el histograma.
# 5. Repetir el experimento de los dos puntos anteriores con dos muestras aleatorias más generadas con los mismos parámetros.
# ¿Qué conclusión saca?
# 
# Parte 3: Convergencia
# El propósito de esta sección es ver en forma práctica los resultados de los teoremas de convergencia.
# 1. Generar cuatro muestras de números aleatorios de tamaño 100, todas con distribución binomial con p = 0,40 y n = 10, n = 20,
# n = 50 y n = 100 respectivamente. Graficar sus histogramas. ¿Qué observa?
# 2. Elija la muestra de tamaño 200 y calcule la media y desviación estándar muestral. Luego, normalice cada dato de la muestra
# y grafique el histograma de la muestra normalizada. Justifique lo que observa.
# 3. Para cada una de las muestras anteriores, calcule la media muestral. Justifique lo que observa.
# 
# Parte 4: Estadística inferencial
# Para terminar, vamos a hacer inferencia con las muestras que generamos y obtener así información sobre sus distribuciones.
# 1. Generar dos muestras N(100, 5), una de tamaño n = 10 y otra de tamaño n = 30. Obtener estimaciones puntuales de su media
# y varianza.
# 2. Suponga que ya conoce el dato de que la distribución tiene varianza 5. Obtener intervalos de confianza del 95% y 98% para
# la media de ambas muestras.
# 3. Repita el punto anterior pero usando la varianza estimada s^2, para la muestra de tamaño adecuado.
# 4. Probar a nivel 0,99 la hipótesis de que la varianza sea σ^2 > 5. Calcular la probabilidad de cometer error tipo II 
# para la hipótesisalternativa σ^2 = 6.
# 5. Agrupando los datos en subgrupos de longitud 0,5, probar a nivel 0,99 la hipótesis de que la muestra proviene 
# de una distribución normal.

# In[97]:


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
from scipy.stats import chi2


# In[98]:


# Bibliografia
# https://relopezbriega.github.io/blog/2016/06/29/distribuciones-de-probabilidad-con-python
# https://relopezbriega.github.io/blog/2015/06/27/probabilidad-y-estadistica-con-python/
# Fn Binomial: https://www.aglarick.com/2020/02/15/generacion-de-la-distribucion-binomial-en-python-con-jupyter-y-matplotlib/
# Fn Normal: https://stackoverrun.com/es/q/3329235


# # Parte 1: Simulación

# # Parte 1 Ejercicio 1

# In[99]:


# En esta primera parte, construiremos varios generadores de números aleatorios que usaremos para obtener muestras con distribu-
# ción conocida sobre las que vamos a trabajar posteriormente.

# 1. Utilizando únicamente la función random de su lenguaje (la función que genera un número aleatorio uniforme entre 0 y 1),
# implemente una función que genere un número distribuido Bernoulli con probabilidad p.


# In[100]:


# np.random.uniform(0,1) "la función que genera un número aleatorio uniforme entre 0 y 1"
# De esta forma, devuelve valores equiprobables entre 0 y 1
# Los valores, los voy a generar de esta forma valor = np.random.uniform(0,1) y dps, comparo con el valor de p (prob)

def fn_bernoulli_random(p):
        if np.random.uniform(0,1) > p:
            return 0
        else:
            return 1


# In[101]:


# implemente una función que genere un array de valores distribuido Bernoulli con probabilidad p.
def fn_bernoulli_array(x,p):
    valores = np.zeros((x))
    for i in range(0,x):
        valores[i-1] = fn_bernoulli_random(p)
    return valores
        


# In[102]:


datos_bernoulli = fn_bernoulli_array(100,0.4)
print(datos_bernoulli)
print ("Número + Cantidad de ocurrencias encontradas ==> %s" % collections.Counter(datos_bernoulli))


# # Parte 1 Ejercicio 2

# In[103]:


# 2. Utilizando la función del punto anterior, implemente otra que genere un número binomial con los parámetros n,p.


# In[104]:


def fn_binomial_random(n,p):
    array_valores = fn_bernoulli_array(n,p)
    exitos=[ensayo==1 for ensayo in array_valores] #Cuento los exitos de los Bernoulli anteriores
    return sum(exitos)


# In[105]:


# Implemento funcion para retornar un conjunto de valores aleatorios siguiendo la distribucion binomial
def fn_binomial_array(casos,n,p):
    valores = np.zeros(casos)
    for i in range(0,casos):
        valores[i-1] = fn_binomial_random(n,p)
    return valores
        


# In[106]:


exitos = fn_binomial_random(50, 0.4)
print("%s éxitos de una Binomial con una muestra de %s y probabilidad %s" % (exitos, 50, 0.4))


# In[107]:


datos_binomial = fn_binomial_array(100,6,0.5)
print(datos_binomial)
print ("Número + Cantidad de ocurrencias encontradas ==> %s" % collections.Counter(datos_binomial))


# # Parte 1 Ejercicio 3

# In[108]:


# Utilizando el procedimiento descrito en el capítulo 6 del Dekking (método de la función inversa o de Monte Carlo), imple-
# mentar una función que permita generar un número aleatorio con distribución Exp(λ).


# Metodo de funcion inversa: pagina 74 del Dekkings

# ![image.png](attachment:image.png) con U(0,1), valores random de distribucion uniforme

# In[109]:


def fn_inversa_exponencial(_lambda, u):
    return -(1/_lambda)* np.log(1-u) 


# In[110]:


def fn_exponencial_random(_lambda):
    numeroRandomConDistribucionUniforme = np.random.uniform(0,1)
    result = fn_inversa_exponencial(_lambda, numeroRandomConDistribucionUniforme)
    return result


# # Parte 1 Ejercicio 4

# In[111]:


# Investigar como generar números aleatorios con distribución normal. Implementarlo.


# Usamos el método de Box-muller para generar numeros random siguiendo una distribucion normal 
# https://es.wikipedia.org/wiki/M%C3%A9todo_de_Box-Muller

# In[242]:


def fn_normal_random(mean, desviacion_estandar):
    theta = 2 * math.pi * np.random.uniform(0,1) 
    rho = math.sqrt(-2 * np.log(1 - np.random.uniform(0,1))) 
    scale = desviacion_estandar * rho
    y = mean + scale * math.sin(theta) 
    return y


# In[243]:


def fn_normal_array(casos, media, desviacion_estandar):
    valores = np.zeros(casos)
    for i in range(0,casos):
        valores[i-1] = fn_normal_random(media, desviacion_estandar)
    return valores


# In[244]:


datos_normal = fn_normal_array(30,100, 5)
print(datos_normal)
print(np.mean(datos_normal))


# # Parte 2: Estadística descriptiva

# # Parte 2 Ejercicio 1

# In[115]:


# Generar tres muestras de números aleatorios Exp(0,5) de tamaño n = 10, n = 30 y n = 200. Para cada una, computar la media
# y varianza muestral. ¿Qué observa?


# Media Muestral ![xn.JPG](attachment:xn.JPG)

# Varianza Muestral ![vm.JPG](attachment:vm.JPG)

# In[116]:


def media_muestral(valores):
    suma = 0
    for i in range(0, len(valores)):
        suma += valores[i]
    
    return suma/len(valores)

def varianza_muestral(valores):
    media = media_muestral(valores)
    
    suma = 0
    for i in range(0, len(valores)):
        suma += math.pow(valores[i]-media, 2)

    varianza = suma/(len(valores)-1)
    return varianza


# In[117]:


def fn_generador_de_muestras_numeros_random_con_dist_exponencial(n,_lambda):
    valores = np.zeros(n)
    for i in range(0,n):
        valores[i] = fn_exponencial_random(_lambda)
    
    #Calculo de media_muestral
    rt_media_muestral = media_muestral(valores)
    
    #Calculo de varianza_muestral
    rt_varianza_muestral = varianza_muestral(valores)

    return valores, rt_media_muestral, rt_varianza_muestral


# Primera muestra con n=10: 

# In[118]:


val_n10, media, varianza = fn_generador_de_muestras_numeros_random_con_dist_exponencial(10,0.5)
#print(val_n10)
#collections.Counter(val_n10) 
print("Media=", media)
print("Varianza=", varianza)


# Segunda muestra con n=30: 

# In[119]:


val_n30, media, varianza = fn_generador_de_muestras_numeros_random_con_dist_exponencial(30,0.5)
#print(val_n30)
#collections.Counter(val_n30) 
print("Media=", media)
print("Varianza=", varianza)


# Segunda muestra con n=200:

# In[120]:


val_n200, media, varianza = fn_generador_de_muestras_numeros_random_con_dist_exponencial(200,0.5)
#print(val_n200)
#collections.Counter(val_n200) 
print("Media=", media)
print("Varianza=", varianza)


# In[121]:


# OBSERVACION: Conforme aumente la cantidad de muestra, aumenta la varianza.


# # Parte 2 Ejercicio 2

# In[122]:


# Para las tres muestras anteriores, graficar los histogramas de frecuencias relativas con anchos de banda 0,4, 0,2 y 0,1; es decir,
# un total de 9 histogramas. ¿Qué conclusiones puede obtener?


# In[123]:


def graficar_frecuencia_relativa(x, res, titulo):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(1, 1, 1)
    
    ax.bar(x, res.frequency, width=res.binsize)
    ax.set_title(titulo)
    ax.set_xlim([x.min(), x.max()])

    plt.show()


# In[124]:


def frecuencia_relativa_con_ancho(conjuntoDeDatos, ancho=0.1, titulo= 'Histograma de frecuencia relativa'):
    
    inicio = int(min(conjuntoDeDatos))
    fin = int(max(conjuntoDeDatos))
    n = int((fin-inicio)/(ancho))
    res = stats.relfreq(conjuntoDeDatos, numbins=n)
    x = res.lowerlimit + np.linspace(0, res.binsize*res.frequency.size,res.frequency.size)
    graficar_frecuencia_relativa(x, res, titulo)


# In[125]:


frecuencia_relativa_con_ancho(val_n10, 0.4)
frecuencia_relativa_con_ancho(val_n10, 0.2)
frecuencia_relativa_con_ancho(val_n10, 0.1)


# In[126]:


frecuencia_relativa_con_ancho(val_n30, 0.4)
frecuencia_relativa_con_ancho(val_n30, 0.2)
frecuencia_relativa_con_ancho(val_n30, 0.1)


# In[127]:


frecuencia_relativa_con_ancho(val_n200, 0.4)
frecuencia_relativa_con_ancho(val_n200, 0.2)
frecuencia_relativa_con_ancho(val_n200, 0.1)


# In[128]:


# OBSERVACIONES: Conforme es mas chico el ancho de banda, se agrupan menos los valores y mas detallada 
# la forma de la curva, disminuyen los valores en eje Y. 


# # Parte 2 Ejercicio 3

# In[129]:


# Generar una muestra de números Bin(10, 0.3) de tamaño de muestra N = 50. Construir la función de distribución empírica de dicha
# muestra.


# In[130]:


# Referencia Funcion de Distribucion Empirica
# https://machinelearningmastery.com/empirical-distribution-function-in-python/#:~:text=An%20empirical%20distribution%20function%20can,specific%20observations%20from%20the%20domain.


# In[131]:


# Referencia sobre Funcion de Distribucion Empirica
# http://halweb.uc3m.es/esp/Personal/personas/jmmarin/esp/Boots/tema2BooPres.pdf


# In[132]:


def funcion_de_distribución_empirica_n(array_de_valores):
    array_de_valores = sorted(array_de_valores)
    ret_value = np.zeros((len(array_de_valores),2))
    
    cantidad = len(array_de_valores)
    
    valor_maximo = max(array_de_valores)
    valor_minimo = min(array_de_valores)
    
    tam_muestra = 0
    
    for i in range(0, cantidad):
        if(array_de_valores[i]<valor_maximo):
            tam_muestra = tam_muestra+1
    
    for i in range(0, tam_muestra):
        ret_value[i,0]=array_de_valores[i]
        ret_value[i,1]=round((i+1)/(tam_muestra+1),2)

    for i in range(tam_muestra, cantidad):
        ret_value[i,0]=array_de_valores[i]
        ret_value[i,1]=1
        
    return ret_value


# In[133]:


casos = 50
n = 10
p = 0.3

array_de_valores = fn_binomial_array(casos, n, p)

distribucion_empirica_ejercicio_3 = funcion_de_distribución_empirica_n(array_de_valores)

print("array de valores con distribucion binomial: " + str(sorted(array_de_valores)))

print("array de valores de F empirica: ",distribucion_empirica_ejercicio_3)


# In[134]:


def frecuencia_relativa(conjuntoDeDatos, ancho_de_barra = 0.1, titulo= 'Histograma de frecuencia relativa'):
    repeticiones = collections.Counter(conjuntoDeDatos)

    total = len(conjuntoDeDatos)
    valores = []
    x = []
    suma = 0
    i = 0
    for clave, valor in repeticiones.items():
        valores.append(valor/total)
        x.append(clave)
        suma += valores[i]
        i += 1
        
    print("suma = ", suma)
    plt.title(titulo)
    plt.bar(x, valores, ancho_de_barra)
    plt.show()


# In[135]:


frecuencia_relativa(array_de_valores)


# In[136]:


def graficar_diagrama_acumulada(valores):
    x, y = zip(*valores)
    plt.step(x,y)
    plt.title('Probabilidad Acumulada F Empirica')
    plt.grid()
    plt.show()


# In[137]:


graficar_diagrama_acumulada(distribucion_empirica_ejercicio_3)


# # Parte 2 Ejercicio 4

# In[138]:


# A partir de la función de distribución empírica del punto anterior, 
# generar una nueva muestra de números aleatorios utilizando
# el método de simulación de la primera parte. 
# Computar la media y varianza muestral y graficar el histograma.


# Ejemplos de boostrap:
# 
# https://datasciencechalktalk.com/2019/11/12/bootstrap-sampling-an-implementation-with-python/
# 
# https://www.linkedin.com/learning/r-para-data-scientist-avanzado/bootstrap-en-r-muestreo?originalSubdomain=es

# In[139]:


def sampleo_bootstrap(array_de_valores,casos):
    len_array = len(array_de_valores)

    sample_mean = []
    for _ in range(casos):
        sample_n = np.random.choice(array_de_valores, size=len_array) #Muestreo con reemplazo
        sample_mean.append(sample_n.mean())

    return sample_mean


# In[140]:


valores = np.zeros(len(distribucion_empirica_ejercicio_3))
for i in range(0, len(distribucion_empirica_ejercicio_3)):
    valores[i] = distribucion_empirica_ejercicio_3[i][1]
    
muestra = sampleo_bootstrap(valores, 50)

print ("Número + Cantidad de ocurrencias encontradas ==> %s" % collections.Counter(sorted(muestra)))
print("\n")

print("media muestral array de valores ejericio 3 = ",
      media_muestral(array_de_valores))
print("media muestral array de valores ejericio 4 = ",
     media_muestral(muestra))

print("varianza muestral array de valores ejericio 3 = ",
      varianza_muestral(array_de_valores))
print("varianza muestral array de valores ejericio 4 = ",
      varianza_muestral(muestra))

frecuencia_relativa(muestra, ancho_de_barra = 0.001, titulo = 'Histograma de nueva muestra')


# # Parte 2 Ejercicio 5

# In[141]:


# Repetir el experimento de los dos puntos anteriores con dos muestras aleatorias más generadas con los mismos parámetros.
# ¿Qué conclusión saca?


# In[142]:


def repeticion_ejercicios_3_y_4():    
    casos = 50
    n = 10
    p = 0.3

    array_de_valores = fn_binomial_array(casos, n, p)
    distribucion_empirica = funcion_de_distribución_empirica_n(array_de_valores)

    valores = np.zeros(len(distribucion_empirica))
    for i in range(0, len(distribucion_empirica)):
        valores[i] = distribucion_empirica[i][1]

    muestra = sampleo_bootstrap(valores, casos)

    print("media muestral array de valores 1 = ",
          media_muestral(array_de_valores))
    print("media muestral array de valores 2 = ",
          media_muestral(muestra))

    print("varianza muestral array de valores 1 = ",
          varianza_muestral(array_de_valores))
    print("varianza muestral array de valores 2 = ",
          varianza_muestral(muestra))

    frecuencia_relativa(muestra, 0.01, titulo = 'Histograma de nueva muestra')


# In[143]:


repeticion_ejercicios_3_y_4()


# 2do conjunto de casos

# In[144]:


repeticion_ejercicios_3_y_4()


# In[145]:


# CONCLUSIÓN: Por un lado, la función escalonada de distribución empirica me muestra los saltos que tendrán cada 
# valor de la muestra respecto de su probabilidad asignada como i/n con i subindice de la muestra y n cantidad de valores.
# Por otro lado, respecto al muestreo de Bootstrap, con este obtenemos conjuntos de datos que representarán la media de cada submuestra.
# Teniendo en cuenta, que la muestra origianl obtenida de forma aleatoria por la distribución, pasará a ser la población.
# Conforme aumentamos la cantidad de casos generados por el muestreo, este aproximará al grafico de una normal y
# los valores obtenidos (que son medias), contendrán al valor de la media muestral original.
# Ademas, observamos que la varianza muestral en la muestra de bootstrap, es mucho mas chica que en el conjunto original.
# Por Teorema Central del Limite, como tiene n>30, y es muy grande, la grafica se asemeja a una distribucion normal.


# # Parte 3: Convergencia

# # Parte 3 Ejercicio 1

# In[146]:


# Generar cuatro muestras de números aleatorios de tamaño 100, todas con distribución binomial con p = 0,40 y n = 10, n = 20,
# n = 50 y n = 100 respectivamente. Graficar sus histogramas. ¿Qué observa?


# In[147]:


val_e1_p3_10 = fn_binomial_array(100,10,0.4)
print(val_e1_p3_10)
frecuencia_relativa(val_e1_p3_10)


# In[148]:


val_e1_p3_20 = fn_binomial_array(100,20,0.4)
print(val_e1_p3_20)
frecuencia_relativa(val_e1_p3_20)


# In[149]:


val_e1_p3_50 = fn_binomial_array(100,50,0.4)
print(val_e1_p3_50)
frecuencia_relativa(val_e1_p3_50)


# In[150]:


val_e1_p3_100 = fn_binomial_array(100,100,0.4)
print(val_e1_p3_100)
frecuencia_relativa(val_e1_p3_100)


# In[151]:


# CONCLUSION: Teniendo en cuenta el p=0.4, puedo estimar que el pico de frencuencia en cada uno, va a estar determinado por el 
# valor del rango que sea la esperanza (n*p) aproximadamente. 


# #  Parte 3 Ejercicio 2

# In[152]:


# Elija la muestra de tamaño 200 y calcule la media y desviación estándar muestral. Luego, normalice cada dato de la muestra
# y grafique el histograma de la muestra normalizada. Justifique lo que observa.


# In[195]:


val_e2_p3_200 = fn_binomial_array(200,10,0.4)
val_e2_p3_200 = sorted(val_e2_p3_200)

print(val_e2_p3_200)

media_muestral_p3_2 = media_muestral(val_e2_p3_200)
print("media muestral: ",media_muestral_p3_2)

varianza_p3_200 = varianza_muestral(val_e2_p3_200)
desviacion_estandar = math.sqrt(varianza_p3_200)
print("desviacion estandar:", desviacion_estandar)


# In[196]:


normal = np.zeros(200)

for i in range(0,200):
    valor = val_e2_p3_200[i]
    calculo_a = valor - media_muestral_p3_2
    calculo = calculo_a / varianza_p3_200
    normal[i] = calculo

df = pd.DataFrame({"valor":val_e2_p3_200,"estandarizado":normal})
print(df)
frecuencia_relativa(normal)


# In[159]:


# En el primer conjunto, la esperanza de la binomial, me da que en el valor 4 tendre el pico de frencuencia de ocurrencia
# Cuando normalizo los valores a un intervalo {0,1}, el valor representado de 4 es 0,5 esto en el histograma
# me muestra el valor con mayor frecuencia de ocurrencia.


# #  Parte 3 Ejercicio 3

# In[160]:


# Para cada una de las muestras anteriores, calcule la media muestral. Justifique lo que observa.


# In[161]:


print("media muestral ejercicio 1 N=100 n=10 = ",media_muestral(val_e1_p3_10))
print("media muestral ejercicio 1 N=100 n=20 = ",media_muestral(val_e1_p3_20))
print("media muestral ejercicio 1 N=100 n=50 = ",media_muestral(val_e1_p3_50))

print("media muestral ejercicio 1 N=100 n=100= ",media_muestral(val_e1_p3_100))
print("Esperanza n=100 = ",0.4*100)

print("media muestral ejercicio 2 N=200 n=10 = ",media_muestral(val_e2_p3_200))
print("Esperanza n=10 = ",0.4*10)


# In[162]:


# Al modificar el parametro n de las distribuciones, estoy modificando la cantidad de éxitos que voy 
# a tener con probabilidad 0.4
# Ademas, en el último caso, aproximo mejor el valor de esperanza de binomial, respecto al 
# primero, ya que estoy tomando mas casos.


# # Parte 4: Estadística inferencial

# #  Parte 4 Ejercicio 1

# In[163]:


# Generar dos muestras N(100, 5), una de tamaño n = 10 y otra de tamaño n = 30. Obtener estimaciones puntuales de su media
# y varianza.


# In[245]:


val_e1_p4_10 = fn_normal_array(10,100, 5)
print(val_e1_p4_10)

media_muestral_n10 = media_muestral(val_e1_p4_10)
print("Media muestral n=10: ",media_muestral_n10)

print("Varianza muestral: ", varianza_muestral(val_e1_p4_10))


# In[246]:


val_e1_p4_30 = fn_normal_array(30,100, 5)
print(val_e1_p4_30)

media_muestral_n30 = media_muestral(val_e1_p4_30)
print("Media muestral n=30: ", media_muestral_n30)

varianza_muestral_n30 = varianza_muestral(val_e1_p4_30)
print("Varianza muestral: ", varianza_muestral_n30)


# # Parte 4 Ejercicio 2

# In[166]:


# Suponga que ya conoce el dato de que la distribución tiene varianza 5.
# Obtener intervalos de confianza del 95% y 98% para la media de ambas muestras.


# Intervalo de confianza con varianza conocida ![intervalo.png](attachment:intervalo.png)

# In[247]:


def con_varianza_conocida(varianza, n, confianza, mu):
    
    sigma = math.sqrt(varianza)
    
    alfa = 1 - confianza
    
    alfa_sobre_2 = alfa / 2
    
    z_alfa_sobre_2 = stats.norm.ppf(alfa_sobre_2) #Busco el valor de la normal en tabla
    
    x_raya = mu #media muestral
    
    limite_inferior = x_raya + (z_alfa_sobre_2*(sigma/math.sqrt(n)))
    limite_superior = x_raya - (z_alfa_sobre_2*(sigma/math.sqrt(n)))
    return limite_inferior, limite_superior


# In[248]:


mu = media_muestral_n10 #N(100,5) n=10
varianza = 5
n = 10 # Como es poblacion normal, puede usarse la formula

print("Valor de mu:",mu)

confianza = 0.95
limite_inferior, limite_superior = con_varianza_conocida(varianza, n, confianza, mu)
print("Intervalo de confianza del 95% se encuentra entre: " + str(limite_inferior) + " <= mu <= " + str(limite_superior))

confianza = 0.98
limite_inferior, limite_superior = con_varianza_conocida(varianza, n, confianza, mu)
print("Intervalo de confianza del 98% se encuentra entre: " + str(limite_inferior) + " <= mu <= " + str(limite_superior))


# In[250]:


mu = media_muestral_n30 #N(100,5) n=30
varianza = 5
n = 30

print("Valor de mu:",mu)

confianza = 0.95
limite_inferior, limite_superior = con_varianza_conocida(varianza, n, confianza, mu)
print("Intervalo de confianza del 95% se encuentra entre: " + str(limite_inferior) + " <= mu <= " + str(limite_superior))

confianza = 0.98
limite_inferior, limite_superior = con_varianza_conocida(varianza, n, confianza, mu)
print("Intervalo de confianza del 98% se encuentra entre: " + str(limite_inferior) + " <= mu <= " + str(limite_superior))


# # Parte 4 Ejercicio 3

# In[170]:


# Repita el punto anterior pero usando la varianza estimada s² , para la muestra de tamaño adecuado.


# Intervalo de confianza con varianza desconocida ![xraya.JPG](attachment:xraya.JPG)

# Valor de n para tamaño de muestra ![valor_n.jpeg](attachment:valor_n.jpeg)

# Error de Muestreo ![SE.JPG](attachment:SE.JPG)

# In[251]:


def obtener_s(valores):
    s2 = varianza_muestral(valores)
    return math.sqrt(s2)

def calcular_n(zeta_alfa,sigma,e):
    return math.pow((zeta_alfa*sigma)/e,2)

def con_varianza_desconocida(N,confianza, valores, mu):
    
    x_raya = mu #media muestral
    
    alfa = 1 - confianza
    alfa_sobre_2 = (alfa / 2)
    
    s = obtener_s(valores) #desvio estandar
    
    zeta_alfa = -round(stats.norm.ppf(alfa_sobre_2),2) #Busco el valor de la normal en tabla
    
    error_estandar = (s / math.sqrt(N)) #zeta_alfa*
    
    n_muestra = int(calcular_n(zeta_alfa,s,error_estandar))
    
    print("n Muestra:",n_muestra , " > ",N)
    
    t_alfa_sobre_2 = stats.t.ppf(alfa_sobre_2, n_muestra-1) #Busco el valor de la T-Student en tabla
    
    limite_inferior = x_raya + (t_alfa_sobre_2*(s/math.sqrt(n_muestra)))
    limite_superior = x_raya - (t_alfa_sobre_2*(s/math.sqrt(n_muestra)))
    
    return limite_inferior, limite_superior


# In[252]:


N = 10
valores = val_e1_p4_10 #fn_normal_array(10,100, 5)
mu = media_muestral_n10 #100

print("Valor de mu:",mu)

confianza = 0.95
limite_inferior, limite_superior = con_varianza_desconocida(N,confianza, valores, mu)
print("Intervalo de confianza del 95% se encuentra entre: "+ str(limite_inferior) + " <= mu <= " + str(limite_superior))

confianza = 0.98
limite_inferior, limite_superior = con_varianza_desconocida(N,confianza, valores, mu)
print("Intervalo de confianza del 98% se encuentra entre: "+ str(limite_inferior) + " <= mu <= " + str(limite_superior)) 


# In[253]:


N = 30
valores = val_e1_p4_30 #fn_normal_array(10,100, 5)
mu = media_muestral_n30 #100

print("Valor de mu:",mu)

confianza = 0.95
limite_inferior, limite_superior = con_varianza_desconocida(N, confianza, valores, mu)
print("Intervalo de confianza del 95% se encuentra entre: "+ str(limite_inferior) + " <= mu <= " + str(limite_superior))

confianza = 0.98
limite_inferior, limite_superior = con_varianza_desconocida(N, confianza, valores, mu)
print("Intervalo de confianza del 98% se encuentra entre: "+ str(limite_inferior) + " <= mu <= " + str(limite_superior))


# # Parte 4 Ejercicio 4

# In[174]:


# Probar a nivel 0,99 la hipótesis de que la varianza sea σ² > 5. Calcular la probabilidad de cometer 
# error tipo II para la hipótesis alternativa σ² = 6.


# Tipos de Error ![WhatsApp%20Image%202020-12-02%20at%2020.32.43.jpeg](attachment:WhatsApp%20Image%202020-12-02%20at%2020.32.43.jpeg)

# Calculo de Z ![Z.JPG](attachment:Z.JPG)

# ![download.png](attachment:download.png)

# In[254]:


def verificar_hipotesis_alternativa(chi_cuadrado_tabla, chi_cuadrado_calculado):
    return chi_cuadrado_calculado > chi_cuadrado_tabla


# In[255]:


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


# In[256]:


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


# In[257]:


# Calcular la probabilidad de cometer error tipo II para la hipótesis alternativa σ² = 6. Para n = 10
n = 10
varianza_nueva = 6
grados_libertad = n - 1

chi_cuadrado_tabla = 21.666 # se obtuvo por tabla para alfa = 0.01 y grados de libertad = 9
s_cuadrado_l = (chi_cuadrado_tabla * varianza) / grados_libertad
chi_cuadrado_calculado = (grados_libertad * s_cuadrado_l) / varianza_nueva

beta = stats.chi2.cdf(chi_cuadrado_calculado, grados_libertad)
print("La probabilidad de cometer error tipo II para n = 10 es:", beta)


# In[258]:


# Calcular la probabilidad de cometer error tipo II para la hipótesis alternativa σ² = 6. Para n = 30
n = 30
varianza_nueva = 6
grados_libertad = n - 1

chi_cuadrado_tabla = 49.588 # se obtuvo por tabla para alfa = 0.01 y grados de libertad = 29
s_cuadrado_l = (chi_cuadrado_tabla * varianza) / grados_libertad
chi_cuadrado_calculado = (grados_libertad * s_cuadrado_l) / varianza_nueva

beta = stats.chi2.cdf(chi_cuadrado_calculado, grados_libertad)
print("La probabilidad de cometer error tipo II para n = 30 es:", beta)


# # Parte 4 Ejercicio 5

# In[259]:


# Agrupando los datos en subgrupos de longitud 0,5, probar a nivel 0,99 la hipótesis de que la muestra proviene 
# de una distribución normal.


# In[260]:


# Partiendo del ejercicio 1 sabemos que sigue una distribución N(100, 5) para la muestra de n = 30

def segmentador_(valores, ancho):
    tamanio = len(valores)
    resultado = []
    final = 0
    contador = 0
    
    inicio = valores[0]
    fin = valores[0] + ancho

    while final < len(valores):
        for i in range(final, tamanio):
            if(valores[i] < fin):
                contador += 1
        
        if (contador >= 5):
            subgrupo = valores[final : (final + contador)]
            resultado.append(subgrupo)
            final += contador
            contador = 0
            inicio = max(subgrupo)
            fin = max(subgrupo) + ancho
        else:
            fin += ancho
    
    tamanio_resultado = len(resultado)
    resultado_final = []
    for i in range(0, tamanio_resultado):
        if (len(resultado[i]) >= 5):
            resultado_final.append(resultado[i])
        else:
            resultado_final[i-1].extend(resultado[i])
    return resultado_final

n = 30
ancho = 0.5

val_e1_p4_30 = sorted(val_e1_p4_30)
conjunto = segmentador(val_e1_p4_30, 0.5)


# In[279]:


def segmentador(valores, ancho):
    valores = sorted(valores)
    tamanio = len(valores)
    resultado = []
    final = 0
    contador = 0
    
    inicio = valores[0]
    fin = valores[0] + ancho

    while final < len(valores):
        for i in range(final, tamanio):
            if(valores[i] < fin):
                contador += 1
        
        if (contador >= 5):
            subgrupo = valores[final : (final + contador)]
            resultado.append(subgrupo)
            final += contador
            contador = 0
            inicio = max(subgrupo)
            fin = max(subgrupo) + ancho
        else:
            fin += ancho
    
    tamanio_resultado = len(resultado)
    resultado_final = []
    for i in range(0, tamanio_resultado):
        if (len(resultado[i]) >= 5):
            resultado_final.append(resultado[i])
        else:
            resultado_final[i-1].extend(resultado[i])
    return resultado_final

n = 36
ancho = 0.5
mu = 75.56
sigma = 14.4
val_e1_p4_30 = [51,70,87,79,54,65,76,91,64,78,73,100,86,58,81,92,77,68,55,99,90,71,52,89,67,50,82,104,88,59,83,75,84,57,72,80]
conjunto = [[50, 51, 52, 54, 55, 57, 58, 59, 64], [65,67, 68, 70, 71, 72, 73],[75, 76, 77, 78, 79, 80, 81, 82, 83, 84],[ 86, 87, 88, 89, 90, 91, 92, 99, 100, 104]]


# In[282]:


n = 30
ancho = 0.5
inicio = int(min(val_e1_p4_30))
fin = int(max(val_e1_p4_30))
n_bins = int((fin-inicio) / ancho)

h_nula =        "la muestra proviene de una distribución normal"
h_alternativa = "la muestra no proviene de una distribución normal"

estadistico_observado = 0

for i in range(0, n_bins):
    fin = inicio + 0.5
    frecuencia_observada = 0
    for j in range(0, len(val_e1_p4_30)):
        if (val_e1_p4_30[j] > inicio and val_e1_p4_30[j] <= fin):
            frecuencia_observada += 1

    if(i == 0):
        zeta = (fin - 100) / math.sqrt(5)
        probabilidad_intervalo = stats.norm.cdf(zeta)
    else:
        zeta_1 = (inicio - 100) / math.sqrt(5)
        zeta_2 = (fin - 100) / math.sqrt(5)
        probabilidad_intervalo = stats.norm.cdf(zeta_2) - stats.norm.cdf(zeta_1)

    frecuencia_esperada = probabilidad_intervalo * n
    resta = frecuencia_observada - frecuencia_esperada
    estadistico_observado += math.pow(resta, 2) / frecuencia_esperada
    inicio = fin

estadistico_teorico = 27.688 # grados de libertad = 13 y alfa = 0.01
es_menor = estadistico_observado < estadistico_teorico

if (es_menor):
    print("Conclusión: hay evidencia para concluir que h nula se cumple.")
else:
    print("Conclusión: no hay evidencia para concluir que h nula se cumple.")


# In[ ]:





# In[ ]:




