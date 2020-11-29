from ejercicios.parte_1_simulacion.normal import fn_normal_array
from funciones.funciones import esperanza_muestral
from funciones.funciones import varianza_muestral

# # Parte 4
# # Ejercicio 1
# Generar dos muestras N(100, 5), una de tamaño n = 10 y otra de tamaño n = 30.
# Obtener estimaciones puntuales de su media y varianza.


val_e1_p4_10 = fn_normal_array(10,100, 5)
print(val_e1_p4_10)
print("Esperanza: " + str(esperanza_muestral(val_e1_p4_10)))
print("Varianza: " + str(varianza_muestral(val_e1_p4_10)))


val_e1_p4_30 = fn_normal_array(30,100, 5)
print(val_e1_p4_30)
print("Esperanza: " + str(esperanza_muestral(val_e1_p4_30)))
print("Varianza: " + str(varianza_muestral(val_e1_p4_30)))
