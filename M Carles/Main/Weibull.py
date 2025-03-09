# GENERAR (N) VELOCIDADES DE VIENTO siguiendo una DISTRIBUCIÓN de WEIBULL y calcula sus PROBABILIDADES asociadas

 # Objetivo:
 # Simular la variabilidad del viento en un parque eólico de manera realista, en lugar de asumir una velocidad constante





# 1. IMPORTAR HERRAMIENTAS

import numpy as np                                                          # Importa la biblioteca NumPy para cálculos numéricos
import matplotlib.pyplot as plt                                             # Biblioteca para configuraciones avanzadas de gráficos
import seaborn as sns                                                       # Para mejorar gráficas
import matplotlib.patheffects as path_effects                               # Para mejorar la visibilidad del texto
from scipy.special import gamma





# 2. WEIBULL DISTRIBUTION

# VIENTOS

# Generación de VIENTOS
def generate_wind_speeds_weibull(                                           # Genera N valores de velocidad del viento siguiendo una distribución de Weibull
        N,                                                                  # N: Número de valores a generar
        k,                                                                  # k: Parámetro de forma (define la dispersión DE VALORES)
        lambda_,                                                            # lambda_: Parámetro de escala (relacionado con la velocidad media)
        season=None,                                                        # season: Temporada del año ('verano', 'invierno'... posibilidad de añadir más) para ajustar los parámetros
        seed=None                                                           # seed: Semilla para reproducibilidad -Permite reproducir los mismos resultados al fijar una semilla aleatoria
        ):                              

    if seed is not None:                                
        np.random.seed(seed)                                
    # Ajuste de parámetros según la estación                                
    if season == 'invierno':                                                # Invierno: más viento por efecto del Mistral
        lambda_ *= 1.2                                                      # Aumentamos la velocidad media
    elif season == 'verano':                                                # Verano: menos viento
        lambda_ *= 0.8                                                      # Reducimos la velocidad media    

    # Generar velocidades de viento correctamente escaladas
    wind_speeds = (np.random.weibull(k, N) * lambda_) 
    # Calcular la función de densidad de probabilidad de Weibull - FÓRMULA MATEMÁTICA
    probabilities = (k / lambda_) * (wind_speeds / lambda_)**(k - 1) * np.exp(-(wind_speeds / lambda_)**k)
    # Normalizar para que las probabilidades sumen 1
    probabilities /= np.sum(probabilities)

    return wind_speeds, probabilities
# return: Lista de velocidades de viento y sus probabilidades asociadas

# DATOS DEL VIENTO
# Parámetros base para la costa mediterránea
N = 10000                                                                   # Más valores para mejor representación estadística
k = 2                                                                       # Parámetro de forma estándar para Weibull - Un estudio sugiere un valor de k=2 para ciertas regiones costeras
lambda_ = 8                                                                 # Velocidad media del viento en m/s - Relacionado con la velocidad media del viento. En el mismo estudio, se encontró un valor de λ=6.77 m/s
season = 'invierno'                                                         # Cambiar a 'verano' o 'invierno' según el caso
seed = 42

# Parámetros del Parque 
n_turbines = 36                                                             # Número total de turbinas en el parque
rated_power = 500

# Generar velocidades de viento
wind_speeds, wind_probs = generate_wind_speeds_weibull(N, k, lambda_, seed)

# Calcular la media teórica esperada
expected_mean = lambda_ * gamma(1 + 1/k)
print(f"Media esperada de la distribución Weibull: {expected_mean:.2f} m/s")

# Verificar eliminación de valores menores a 3 m/s
filtered_wind_speeds = [v for v in wind_speeds if v >= 3]
mean_filtered_wind_speed = np.mean(filtered_wind_speeds)
print(f"Media tras eliminar valores <3 m/s: {mean_filtered_wind_speed:.2f} m/s")


# POTENCIA

 # Convierte velocidades de viento en potencia generada usando la ecuación de potencia eólica
def wind_speed_to_power(                                                   
        wind_speeds,                                                        # wind_speeds: Lista de velocidades de viento en m/s
        rho=1.225,                                                          # rho: Densidad del aire en kg/m³ (por defecto 1.225)
        Cp=0.45,                                                            # Cp: Coeficiente de potencia de la turbina (por defecto 0.45)
        A=38000,                                                            # Área del rotor en m² (por defecto 38000, turbina offshore grande) - (Siemens Gamesa SG 14-222 DD)
        n_turbines=36,                                                      # Número total de turbinas en el parque
        ):
    return (0.5 * rho * A * Cp * (wind_speeds ** 3))*n_turbines             # Fórmula Potencia en función de Velocidad
# Calcular la potencia generada para cada velocidad de viento en el parque
power_generated = wind_speed_to_power(wind_speeds, n_turbines=n_turbines)


# FP - Factor de Potencia
 #     Calcula el Factor de Capacidad (FC) del parque eólico.
def calculate_capacity_factor(mean_power_generated, rated_power):
    return (mean_power_generated / rated_power) * 100                       # Devuelve el FP en %





# 4. CÁLCULOS ADICIONALES PARA LAS GRÁFICAS

# Calcular la velocidad media y la mediana 
mean_wind_speed = np.mean(wind_speeds)
median_wind_speed = np.median(wind_speeds)
# Calcular la potencia media y la mediana 
mean_power_generated = np.mean(power_generated)
median_power_generated = np.median(power_generated)
capacity_factor = calculate_capacity_factor(mean_power_generated / 1e6, rated_power)


# Velocidad del viento asumida por Carles
P_parque = 500e6                                                            # Potencia en W (500 MW) del parque asumido por Carles
n_turbinas = 36                                                             # Estimamons un total de 36 molinos para obtener la P de cada turbina - asumimos el modelo Siemens Gamesa SG 14-222 DD (14 MW por turbina), que es una de las opciones más utilizadas para parques offshore
P_turbina = P = P_parque / n_turbinas                                           # Potencia por cada turbina
rho = 1.225                                                                 # Densidad del aire en kg/m³
Cp = 0.4                                                                    # Coeficiente de potencia
A = 38700                                                                   # Área del rotor en m² (Siemens Gamesa SG 14-222 DD)
v_carles = ((2 * P) / (rho * A * Cp)) ** (1/3)                              # Cálculo de la velocidad del viento
print("Velocidad del viento asumida por Carles",v_carles)                   # Velocidad del viento en m/s
# potancia asumida por Carles
P_carles = 500e6





# 5. GRÁFICAS

# Mostrar resultadoS en el terminal 
#print("Velocidades de viento generadas:", wind_speeds)
print("Probabilidades asociadas:", wind_probs)
print("Suma de probabilidades (debe ser 1):", np.sum(wind_probs))
print(f"Factor de Potencia (FP): {capacity_factor:.2f}%")



# VELOCIDADES DEL VIENTO
 # Visualización de velocidades de viento
# Crea lienzo donde se dibujará el gráfico
plt.figure(figsize=(10,6))
     # figsize=(10,6) establece el tamaño de la figura en pulgadas (10 de ancho, 6 de alto)
# Muestra un histograma de los valores de potencia generada
sns.histplot(wind_speeds, bins=30, kde=True, color='royalblue', alpha=0.6, label='Distribución generada')
     # power_generated / 1e6: Convierte los valores de potencia de vatios (W) a megavatios (MW)
     # bins=100: Divide el eje X en 100 intervalos (aumenta o disminuye el detalle del histograma)
     # kde=True: Activa la Curva de Densidad que suaviza la distribución
     # color='darkorange': Color de las barras
     # alpha=0.6: Transparencia del histograma (1=opaco, 0=transparente)
     # label='Distribución generada': Texto que aparecerá en la leyenda 
# plt.plot(): No dibuja nada, pero agrega texto en la leyenda
plt.axvline(mean_wind_speed, color='red', linestyle='--', label=f'Media: {mean_wind_speed:.2f} m/s')
# plt.axvline(): Dibuja líneas verticales en el gráfico para marcar valores de referencia
plt.axvline(median_wind_speed, color='green', linestyle='-.', label=f'Mediana: {median_wind_speed:.2f} m/s')
plt.axvline(v_carles, color='purple', linestyle=':', label=f'Viento Carles: {v_carles:.2f} m/s')
     # power_generated / 1e6: Convierte los valores de potencia de vatios (W) a megavatios (MW)
     # color: Color de la línea
     # linestyle: Tipo de línea - '--' 
     # label: Texto de la leyenda
# Etiquetas de los ejes X e Y     
plt.xlabel('Velocidad del viento (m/s)', fontsize=14)
plt.ylabel('Densidad de probabilidad', fontsize=14)
# Título del gráfico
plt.title(f'Distribución de Weibull - {season.capitalize()} en la Costa Mediterránea', fontsize=16, fontweight='bold')
     # Usa season.capitalize() para poner en mayúscula la primera letra de la estación (ejemplo: "Invierno")
     # fontweight='bold': Negrita
# Muestra la leyenda del gráfico
plt.legend(fontsize=12)
     # True: Activa la cuadrícula
     # linestyle='--': Estilo de línea discontinua
     # alpha=0.7: Transparencia de la cuadrícula
# Ajusta el rango de valores en el eje X
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()



# POTENCIA GENERADA - 2000 MW
 # Visualización de potencia generada - (0-2000)MW
# Crea lienzo donde se dibujará el gráfico
plt.figure(figsize=(10,6))                                                                                                   
     # figsize=(10,6) establece el tamaño de la figura en pulgadas (10 de ancho, 6 de alto)
# Muestra un histograma de los valores de potencia generada
sns.histplot(power_generated / 1e6, bins=100, kde=True, color='darkorange', alpha=0.6, label='Distribución generada')       
     # power_generated / 1e6: Convierte los valores de potencia de vatios (W) a megavatios (MW)
     # bins=100: Divide el eje X en 100 intervalos (aumenta o disminuye el detalle del histograma)
     # kde=True: Activa la Curva de Densidad que suaviza la distribución
     # color='darkorange': Color de las barras
     # alpha=0.6: Transparencia del histograma (1=opaco, 0=transparente)
     # label='Distribución generada': Texto que aparecerá en la leyenda 
# plt.plot(): No dibuja nada, pero agrega texto en la leyenda
plt.plot([], [], ' ', label=f'FP : {capacity_factor:.2f}%')
# plt.axvline(): Dibuja líneas verticales en el gráfico para marcar valores de referencia
plt.axvline(mean_power_generated / 1e6, color='red', linestyle='--', label=f'Media: {mean_power_generated / 1e6:.2f} MW')
plt.axvline(median_power_generated / 1e6, color='green', linestyle='-.', label=f'Mediana: {median_power_generated / 1e6:.2f} MW')
plt.axvline(500, color='purple', linestyle=':', label=f'Potencia Carles: 500.00 MW')
     # power_generated / 1e6: Convierte los valores de potencia de vatios (W) a megavatios (MW)
     # color: Color de la línea
     # linestyle: Tipo de línea - '--' 
     # label: Texto de la leyenda
# Etiquetas de los ejes X e Y     
plt.xlabel('Potencia generada (MW)', fontsize=14)
plt.ylabel('Densidad de probabilidad', fontsize=14)
# Título del gráfico
plt.title(f'Distribución de Potencia Generada - {season.capitalize()} en la Costa Mediterránea', fontsize=16, fontweight='bold')
     # Usa season.capitalize() para poner en mayúscula la primera letra de la estación (ejemplo: "Invierno")
     # fontweight='bold': Negrita
# Muestra la leyenda del gráfico
plt.legend(fontsize=12, frameon=True)
     # frameon=True: Muestra un cuadro alrededor de la leyenda
# Activa la cuadrícula en el gráfico
plt.grid(True, linestyle='--', alpha=0.7)
     # True: Activa la cuadrícula
     # linestyle='--': Estilo de línea discontinua
     # alpha=0.7: Transparencia de la cuadrícula
# Ajusta el rango de valores en el eje X
plt.xlim(0, 2000) 
plt.show()



# POTENCIA GENERADA - 5000MW
 # Visualización de potencia generada - (0-5000)MW
# Crea lienzo donde se dibujará el gráfico
plt.figure(figsize=(10,6))                                                                                                   
     # figsize=(10,6) establece el tamaño de la figura en pulgadas (10 de ancho, 6 de alto)
#  Muestra un histograma de los valores de potencia generada
sns.histplot(power_generated / 1e6, bins=100, kde=True, color='darkorange', alpha=0.6, label='Distribución generada')       
     # power_generated / 1e6: Convierte los valores de potencia de vatios (W) a megavatios (MW)
     # bins=100: Divide el eje X en 100 intervalos (aumenta o disminuye el detalle del histograma)
     # kde=True: Activa la Curva de Densidad que suaviza la distribución
     # color='darkorange': Color de las barras
     # alpha=0.6: Transparencia del histograma (1=opaco, 0=transparente)
     # label='Distribución generada': Texto que aparecerá en la leyenda
# plt.plot(): No dibuja nada, pero agrega texto en la leyenda
plt.plot([], [], ' ', label=f'FP : {capacity_factor:.2f}%')
# plt.axvline(): Dibuja líneas verticales en el gráfico para marcar valores de referencia
plt.axvline(mean_power_generated / 1e6, color='red', linestyle='--', label=f'Media: {mean_power_generated / 1e6:.2f} MW')
plt.axvline(median_power_generated / 1e6, color='green', linestyle='-.', label=f'Mediana: {median_power_generated / 1e6:.2f} MW')
plt.axvline(500, color='purple', linestyle=':', label=f'Potencia Carles: 500.00 MW')
     # power_generated / 1e6: Convierte los valores de potencia de vatios (W) a megavatios (MW)
     # color: Color de la línea
     # linestyle: Tipo de línea - '--' 
     # label: Texto de la leyenda
# Etiquetas de los ejes X e Y     
plt.xlabel('Potencia generada (MW)', fontsize=14)
plt.ylabel('Densidad de probabilidad', fontsize=14)
# Título del gráfico
plt.title(f'Distribución de Potencia Generada - {season.capitalize()} en la Costa Mediterránea', fontsize=16, fontweight='bold')
     # Usa season.capitalize() para poner en mayúscula la primera letra de la estación (ejemplo: "Invierno")
     # fontweight='bold': Negrita
# Muestra la leyenda del gráfico
plt.legend(fontsize=12, frameon=True)
     # frameon=True: Muestra un cuadro alrededor de la leyenda
# Activa la cuadrícula en el gráfico
plt.grid(True, linestyle='--', alpha=0.7)
     # True: Activa la cuadrícula
     # linestyle='--': Estilo de línea discontinua
     # alpha=0.7: Transparencia de la cuadrícula
# Ajusta el rango de valores en el eje X
plt.xlim(0, 5000) 
plt.show()


