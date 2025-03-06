# Este archivo proves_nsga2.py es la versión mejorada de main.py y actúa como el punto de entrada principal para ejecutar la optimización multiobjetivo utilizando el algoritmo NSGA-II




# 1. IMPORTAR HERRAMIENTAS


import numpy as np                                                 # numpy: Librería. Se utiliza para operaciones matemáticas y de álgebra lineal
import matplotlib.pyplot as plt                                    # matplotlib.pyplot: una de las bibliotecas más populares en Python para crear gráficos y visualizaciones. Cuando se importa como plt, se convierte en una convención ampliamente usada para trabajar con gráficos (pyplot proporciona una interfaz similar a la de MATLAB para crear gráficos de forma sencilla)
import matplotlib as mpl                                           # ""
import time                                                        # time: Proporciona varias funciones relacionadas con la medición y el manejo del tiempo (se utiliza para medir la duración de ciertas operaciones)

# pymoo: Biblioteca utilizada para optimización multiobjetivo.
from pymoo.algorithms.moo.nsga2 import NSGA2                       # Algoritmo NSGA-II para optimización multiobjetivo
from pymoo.optimize import minimize                                # minimize: Función para ejecutar el proceso de optimización
from pymoo.visualization.scatter import Scatter                    # Scatter: Para visualizar los resultados

# Importación de operadores genéticos y métodos de inicialización
from pymoo.core.mixed import MixedVariableGA                       # Algoritmo Genético con soporte para variables mixtas
from pymoo.core.variable import Real, Integer, Choice, Binary      # Tipos de variables de decisión
from pymoo.operators.sampling.rnd import FloatRandomSampling       # Método de inicialización para variables continuas
from pymoo.core.mixed import MixedVariableSampling                 # Muestreo para variables mixtas
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival     # Método de supervivencia en NSGA-II basado en ranking y crowding distance
from pymoo.algorithms.moo.nsga2 import RankAndCrowding             # Otro método de selección basado en crowding distance
from pymoo.constraints.as_penalty import ConstraintsAsPenalty      # Penalización de restricciones
from pymoo.decomposition.asf import ASF                            # Método para descomposición de objetivos (Achievement Scalarizing Function)
from pymoo.core.evaluator import Evaluator                         # Evaluador de individuos
from pymoo.core.individual import Individual                       # Representación de un individuo en la población

# Importación de la función de costos y del problema de optimización (Nuestras funciones)
from windopti import MixedVariableProblem
from costac_2 import costac_2

problem = MixedVariableProblem()  # Instancia del problema de optimización





# 2 CREAR MÉTODOS


# MÉTODO 1 - NSGA II

# Configuración del algoritmo NSGA-II con variables mixtas
algorithm = MixedVariableGA(
    pop_size=150,                                             # Tamaño de la población
    sampling=MixedVariableSampling(),                         # Método de muestreo para variables mixtas
    survival=RankAndCrowding(crowding_func="pcd")             # Método de selección basado en crowding distance
)

# Iniciar la optimización y medir el tiempo de ejecución
start_time = time.time()                                      # start_time: Guarda el tiempo de inicio.
res = minimize(                                               # minimize: Ejecuta la optimización
    problem,                                                  # problem = MixedVariableProblem()
    algorithm,                                                # NSGA II
    termination=('n_gen', 6),                                 # Criterio de parada: 6 generaciones
    seed=1,                                                   # Semilla para reproducibilidad
    verbose=True,                                             # Mostrar progreso en terminal
    save_history=True                                         # Guardar el historial de generaciones
)
end_time = time.time()                                        # end_time: Guarda el tiempo de finalización.
execution_time = end_time - start_time                        # calcula la duración
print("Execution time NSGA: ", execution_time, "s")           # Muestra en pantalla el tiempo de ejecución de NSGA II

# Cálculo del mejor punto usando pesos para cada objetivo
weights = np.array([0.5, 0.5])                                # Asignación de pesos a cada objetivo
decomp = ASF()                                                # Inicializa el método Achievement Scalarizing Function (ASF), que evalúa soluciones en función de los pesos.
I = decomp(res.F, weights).argmin()                           # Índice de la mejor solución según ASF


# Imprimir la mejor solución encontrada en función de (X,F,C,I)

# print("Best solution found: \nX = %s\nF = %s\nC = %s" % (   # Muestra la mejor solución encontrada en términos de:
   # res.X,                                                   # X: Variables de decisión óptimas.
   # res.F,                                                   # F: Valores de las funciones objetivo.
   # res.CV))                                                 # C → Restricciones violadas (si hay restricciones).

print("Best solution found weighted: \nX = %s\nF = %s" % (   # Muestra la mejor solución según los pesos definidos previamente (ASF) 
    res.X[I],                                                # res.X[I]: Variables de decisión de la mejor solución ponderada -> Sale: [react1_bi, react2_bi, ... → Estados de activación de los reactores (True/False)] [vol_level → Nivel de voltaje seleccionado] [n_cables → Número de cables en paralelo] [S_rtr → Potencia nominal del transformador] [react1, react2, ... → Valores de los reactores]
    res.F[I]))                                               # res.F[I]: Valores de los objetivos de la mejor solución ponderada -> Sale: [Costo de inversión (M€)] [Costo técnico (M€)]





# MÉTODO 2 - Búsqueda Aleatoria

# Configuración de Búsqueda Aleatoria
start_time2 = time.time()                                                                                    # start_time: Guarda el tiempo de inicio.
trials = 700                                                                                                 # Número de iteraciones de búsqueda aleatoria
ff = costac_2                                                                                                # Función de costo utilizada (creada por nosotros)
d = 13                                                                                                       # Dimensión del problema (cantidad de variables de decisión)
num_int = 7                                                                                                  # Número de variables de decisión que son enteras
lb = np.array([3, 2, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 500e6])                                         # Límite inferior de las variables de descisión
ub = np.array([3, 3, 1, 1, 1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1000e6])                                        # Límite superior de las variables de descisión
p_owf = 5                                                                                                    # Representa la potencia del parque eólico.
random_check = np.zeros((trials, 6))                                                                         # Matriz para almacenar los resultados de costos de cada prueba.
x_history = np.zeros((trials, d))                                                                            # Matriz para registrar las soluciones generadas

# Generación y Evaluación de  700 soluciones aleatorias
for i in range(trials):                                                                                      # Recorre 700 iteraciones (una por cada prueba aleatoria)
    x0 = np.zeros(d)                                                                                         # Recorre 700 iteraciones (una por cada prueba aleatoria)
    x0[:num_int] = np.round(np.random.rand(num_int) * (ub[:num_int] - lb[:num_int]) + lb[:num_int])          # Genera valores aleatorios enteros dentro de los límites definidos (lb y ub)
    x0[num_int:] = np.random.rand(d - num_int) * (ub[num_int:] - lb[num_int:]) + lb[num_int:]                # Genera valores aleatorios continuos dentro de los límites definidos (lb y ub)
    x_history[i, :] = x0                                                                                     # Genera valores aleatorios continuos dentro de los límites
    vol, n_cables, *react, S_rtr = x0                                                                        # Extrae los valores de voltaje, número de cables, reactores y potencia del transformador
    cost_invest, cost_tech, cost_full = ff(vol, n_cables, *react, S_rtr, p_owf)                              # Evalúa la función de costo costac_2 con los valores generados (extraidos)
    random_check[i, :] = [cost_invest, cost_tech, cost_full[10], cost_full[2], cost_full[3], cost_full[11]]  # Guarda los costos calculados en random_check - Incluye inversión (cost_invest), costo técnico (cost_tech), y otros costos específicos extraídos de cost_full

# Encontrar la mejor solución de la búsqueda aleatoria
min_sum_row_index = np.argmin(np.sum(random_check, axis=1))                                                  # Calcula la suma de los costos en cada fila de random_check - Identifica el índice de la fila con el menor costo total
print("Best random search solution:", random_check[min_sum_row_index])                                       # Muestra los valores de costo asociados a la mejor solución aleatoria -> Sale: [Costo de inversión (M€)] [Costo técnico (M€)] [ Penalización por sobrevoltajes] [Penalización por sobrecorrientes] [Penalización por desequilibrio de reactivos] [Penalización por subtensión]
print("Best random search parameters:", x_history[min_sum_row_index, :])                                     # Muestra los valores de decisión (parámetros) de la mejor solución encontrada -> Sale: [Voltaje y nº de cables] [Estado de activación de los reactores (1-Activo 0-Apagado)] [Valores de los reactores] [Potencia nominal del transformador]
end_time2 = time.time()                                                                                      # Captura el tiempo al finalizar la ejecución
execution_time2 = end_time2 - start_time2                                                                    # Calcula el tiempo total
print("Execution time random search: ", execution_time2, "s")                                                # Imprime el tiempo de ejecución en segundos de la Búsqueda Aleatoria





# 3 GRÁFICAS

# GRÀFICA PARETO - MÉTODO 1 - NSAG II - MEJORAR GÁFICA!!!!!!!!!!!!!!!!!!!
  # Graficar el frente de Pareto encontrado 
plot = Scatter()
plot.add(res.F, facecolor="none", edgecolor="black")
plot.add(res.F[I], color="red", s=50)                  # Resaltar el mejor punto
plot.show()


# GRÁFICA COMPARACIÓN - MÉTODO 1 (NSGA II) VS MÉTODO 2 (Búsqueda Aletoria) - MEJORAR GÁFICA!!!!!!!!!!!!!!!!!!!!!!
  # Comparación gráfica entre NSGA-II y búsqueda aleatoria
plt.scatter(random_check[:, 0], random_check[:, 1], facecolor="none", edgecolor="black", label='Random search')   # info: Búsqueda Aleatoria
plt.scatter(res.F[:, 0], res.F[:, 1], facecolor="none", edgecolor="red", label='NSGA-II Pareto Front')            # info: NSGA II
plt.xlabel("Investment cost [M€]")                                                                                # eje X
plt.ylabel("Technical cost [M€]")                                                                                 # eje Y
plt.title("Set of solutions comparing NSGA-II and random search")                                                 # Título
plt.legend()
plt.show()
