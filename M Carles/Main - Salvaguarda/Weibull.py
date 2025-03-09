# GENERAR (N) VELOCIDADES DE VIENTO siguiendo una DISTRIBUCIÓN de WEIBULL y calcula sus PROBABILIDADES asociadas

 # Objetivo:
 # Simular la variabilidad del viento en un parque eólico de manera realista, en lugar de asumir una velocidad constante

import numpy as np                           # Importa la biblioteca NumPy para cálculos numéricos
 

def generate_wind_speeds_weibull(             # Genera N valores de velocidad del viento siguiendo una distribución de Weibull
        N,                                    # N: Número de valores a generar
        k,                                    # k: Parámetro de forma (define la dispersión DE VALORES)
        lambda_,                              # lambda_: Parámetro de escala (relacionado con la velocidad media)
        seed=None                             # seed: Semilla para reproducibilidad -Permite reproducir los mismos resultados al fijar una semilla aleatoria
        ): 
    if seed is not None:
        np.random.seed(seed)
    # Generar velocidades de viento correctamente escaladas
    wind_speeds = (np.random.weibull(k, N) * lambda_) / np.exp(1/k)
    # Calcular la función de densidad de probabilidad de Weibull - FÓRMULA MATEMÁTICA
    probabilities = (k / lambda_) * (wind_speeds / lambda_)**(k - 1) * np.exp(-(wind_speeds / lambda_)**k)
    # Normalizar para que las probabilidades sumen 1
    probabilities /= np.sum(probabilities)

    return wind_speeds, probabilities
# return: Lista de velocidades de viento y sus probabilidades asociadas


# Parámetros de Weibull para prueba

N = 100                                       # Número de muestras
k = 2                                         # Parámetro de forma (mayor k → distribución más uniforme)
lambda_ = 10                                  # Parámetro de escala (mayor lambda_ → mayores velocidades)
seed = 42                                     # Para resultados reproducibles

# Generar velocidades de viento
wind_speeds, wind_probs = generate_wind_speeds_weibull(N, k, lambda_, seed)

# Mostrar resultados
print("Velocidades de viento generadas:", wind_speeds)
print("Probabilidades asociadas:", wind_probs)
print("Suma de probabilidades (debe ser 1):", np.sum(wind_probs))