# Mmain.py
# ---------------------------------------------------------------------------------
# PUNTO DE ENTRADA PRINCIPAL para ejecutar la OPTIMIZACIÓN MULTIOBJETIVO 
# utilizando MixedVariableGA (NSGA-II adaptado a variables mixtas) con Pymoo.
# Se mantiene además la Búsqueda Aleatoria y el método OPF (determinista)
# ---------------------------------------------------------------------------------
from costac_2 import costac_2

import numpy as np
import time
import matplotlib.pyplot as plt

# pymoo imports
from pymoo.optimize import minimize
from pymoo.core.mixed import MixedVariableGA, MixedVariableSampling
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.decomposition.asf import ASF
from pymoo.visualization.scatter import Scatter

# Importa el problema de optimización (definido en windopti.py)
from windopti import MixedVariableProblem







# METODO CARLES 
#import sys
# Asegúrate de que la carpeta donde se encuentran los archivos antiguos esté en el path:
#sys.path.append("carles_old")  # Reemplaza "carles_old" por la ruta correcta

# Importa la versión antigua del problema, renombrándola para diferenciarla
from carles_windopti import MixedVariableProblem as MixedVariableProblemCarles

from pymoo.optimize import minimize
from pymoo.core.mixed import MixedVariableGA, MixedVariableSampling
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.decomposition.asf import ASF
import numpy as np

# Crea una instancia del problema antiguo (Carles)
problem_carles = MixedVariableProblemCarles()

# Configura el algoritmo igual que en tu método actual (puedes ajustar parámetros si es necesario)
algorithm_carles = MixedVariableGA(
    pop_size=150,
    sampling=MixedVariableSampling(),
    survival=RankAndCrowdingSurvival(crowding_func="pcd"),
    seed=1
)

# Ejecuta la optimización antigua (NSGA-II de Carles)start_time = time.time()
start_time = time.time()
res_carles = minimize(
    problem_carles,
    algorithm_carles,
    termination=('n_gen', 6),
    seed=1,
    verbose=True,
    save_history=True
)
end_time = time.time()

print("Execution time MixedVariableGA - Carles:", end_time - start_time, "s")

# Calcula el índice del mejor individuo usando ASF, con los mismos pesos que en tu método actual
weights = np.array([0.5, 0.5])
decomp_carles = ASF()
I_carles_raw = decomp_carles(res_carles.F, weights).argmin()
if hasattr(I_carles_raw, "__len__"):
    I_carles_raw = I_carles_raw[0]
I_carles = int(np.clip(I_carles_raw, 0, res_carles.F.shape[0]-1))

print("Mejor solución del método antiguo (Carles):")
print("X =", res_carles.X[I_carles])
print("F =", res_carles.F[I_carles])

from pymoo.visualization.scatter import Scatter



plt.figure(figsize=(8,6))
# Puntos no óptimos: círculos sin relleno, solo borde negro
plt.scatter(res_carles.F[:, 0], res_carles.F[:, 1], facecolors="none", edgecolors="black", marker="o", s=50, label="NSGA-II (Potencia Fija)")
# Punto óptimo: círculo rojo relleno
plt.scatter(res_carles.F[I_carles, 0], res_carles.F[I_carles, 1], color="red", marker="o", s=80, label="Óptimo")
plt.xlabel("Coste de Inversión [M€]", fontsize=14, labelpad=10)
plt.ylabel("Coste Técnico [M€]", fontsize=14, labelpad=10)
plt.xlim(170, 205)
plt.ylim(0, 1200)
plt.title("NSGA-II (Potencia Fija) Carles - Pareto Front", fontsize=16, fontweight="bold")
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()





# ---------------------------------------------------------------------------------
# MÉTODO 1: Optimización con MixedVariableGA
# ---------------------------------------------------------------------------------

problem = MixedVariableProblem()

# Usamos MixedVariableGA con su muestreo para variables mixtas
algorithm = MixedVariableGA(
    pop_size=300,
    sampling=MixedVariableSampling(),
    survival=RankAndCrowdingSurvival(crowding_func="pcd"),
    seed=1
)

start_time = time.time()
res = minimize(
    problem,
    algorithm,
    termination=('n_gen', 5),  # Usa el mismo número de generaciones que antes
    seed=1,
    verbose=True,
    save_history=True
)
end_time = time.time()
print("Execution time MixedVariableGA:", end_time - start_time, "s")

# Selección de la mejor solución usando ASF (mismos pesos que antes)
weights = np.array([0.5, 0.5])
decomp = ASF()
I = decomp(res.F, weights).argmin()

print("Mejor solución (según ASF con pesos [0.5, 0.5]):")
print("X =", res.X[I])
print("F =", res.F[I])

plt.figure(figsize=(8,6))
# Puntos no óptimos: círculos sin relleno, solo borde negro
plt.scatter(res.F[:, 0], res.F[:, 1], facecolors="none", edgecolors="black", marker="o", s=50, label="NSGA-II (Potencia Fija)")
# Punto óptimo: círculo rojo relleno
plt.scatter(res.F[I, 0], res.F[I, 1], color="red", marker="o", s=80, label="Óptimo")
plt.xlabel("Coste de Inversión [M€]", fontsize=14, labelpad=10)
plt.ylabel("Coste Técnico [M€]", fontsize=14, labelpad=10)
plt.xlim(170, 205)
plt.ylim(0, 1200)
plt.title("NSGA-II (Potencia Fija) - Pareto Front", fontsize=16, fontweight="bold")
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()




# ---------------------------------------------------------------------------------
# MÉTODO 2: Búsqueda Aleatoria (como antes)
# ---------------------------------------------------------------------------------
# MÉTODO 2 - Búsqueda Aleatoria
# Configuración de Búsqueda Aleatoria
start_time2 = time.time()                                                                                    # start_time: Guarda el tiempo de inicio.
trials = 500                                                                                                 # Número de iteraciones de búsqueda aleatoria
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
print("Execution time random search: ", execution_time2, "s") 
# ---------------------------------------------------------------------------------
# MÉTODO 3: OPF - Optimal Power Flow (usando costac_2)
# ---------------------------------------------------------------------------------
# Se recomienda utilizar el módulo costac_2 para calcular los costos OPF
from costac_2 import costac_2  # Asegúrate de que costac_2 está en tu path

# Definición de la solución OPF (igual que antes)
p_owf = 5
x_opf = np.array([
    3,         # vol (interpreta como "vol220" o similar según la lógica original)
    2,         # n_cables
    1,         # react1_bi
    1,         # react2_bi
    0,         # react3_bi
    1,         # react4_bi
    0,         # react5_bi
    0.519,     # react1_val
    0.953,     # react2_val
    0.0,       # react3_val
    0.737,     # react4_val
    0.0,       # react5_val
    509.72e6   # S_rtr
])
vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, \
react1_val, react2_val, react3_val, react4_val, react5_val, S_rtr = x_opf

# Se calcula el costo utilizando costac_2 (como en tu versión original)
cost_invest_opf, cost_tech_opf, cost_fullopf = costac_2(
    vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi,
    react1_val, react2_val, react3_val, react4_val, react5_val, S_rtr, p_owf
)

# Desempaquetado de los costos detallados (según tu código original)
(c_vol, c_curr, c_losses, c_react, cost_tech, c_cab, c_gis, c_tr, c_reac,
 cost_invest, c_volover, c_volunder, c_ss, average_v) = cost_fullopf

print("Resultados OPF:")
print("Coste de Inversión:", cost_invest_opf)
print("Coste Técnico:", cost_tech_opf)
print("Detalle de Costes:", cost_fullopf)



# -------------------------------
# Método 4: NSGA-II con Potencia Variable
# -------------------------------
from var_windopti import MixedVariableProblem as MixedVariableProblemVar
from pymoo.optimize import minimize
from pymoo.core.mixed import MixedVariableGA, MixedVariableSampling
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.decomposition.asf import ASF

# Crear instancia del problema con potencia variable
problem_var = MixedVariableProblemVar()

# Configuración del algoritmo para variables mixtas (igual que en el método fijo, ajusta si es necesario)
algorithm_var = MixedVariableGA(
    pop_size=200,
    sampling=MixedVariableSampling(),
    survival=RankAndCrowdingSurvival(crowding_func="pcd"),
    seed=1
)

# Ejecutar la optimización (puedes ajustar el número de generaciones)
res_var = minimize(
    problem_var,
    algorithm_var,
    termination=('n_gen', 5),
    seed=1,
    verbose=True,
    save_history=True
)

# Seleccionar la mejor solución usando ASF (Achievement Scalarizing Function)
weights = np.array([0.5, 0.5])
decomp = ASF()
# Calculamos I
I_var_raw = decomp(res_var.F, weights).argmin()

# En caso de que devuelva un array (si hay empates), tomar el primer índice:
if hasattr(I_var_raw, "__len__"):
    I_var_raw = I_var_raw[0]

# Asegurar que no exceda el tamaño de la población:
I_var = int(np.clip(I_var_raw, 0, res_var.F.shape[0]-1))

print("I_var =", I_var, "res_var.F.shape=", res_var.F.shape)



print("Best solution with variable power:")
print("X =", res_var.X[I_var])
print("F =", res_var.F[I_var])


plt.figure(figsize=(8,6))
# Puntos no óptimos: círculos sin relleno, solo borde negro
plt.scatter(res_var.F[:, 0], res_var.F[:, 1], facecolors="none", edgecolors="black", marker="o", s=50, label="NSGA-II Variable")
plt.scatter(res_var.F[I_var, 0], res_var.F[I_var, 1], color="red", marker="o", s=80, label="Óptimo Variable")
plt.xlabel("Coste de Inversión [M€]", fontsize=14, labelpad=10)
plt.ylabel("Coste Técnico [M€]", fontsize=14, labelpad=10)
plt.xlim(150, 205)
plt.ylim(0, 1200)
plt.title("NSGA-II (Potencia Variable) - Pareto Front", fontsize=16, fontweight="bold")
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()


print("Fin de la ejecución de Mmain.py")





# 3 GRÁFICAS

# Gráfics que nos muestran las distintas soluciones mediante optimización - Para validar el uso de NSGA

plt.figure(figsize=(10,8))

# NSGA-II Fixed (azul)
plt.scatter(res.F[:, 0], res.F[:, 1], facecolors="none", edgecolors="blue", marker="o", s=50, label="NSGA-II Fixed")
plt.scatter(res.F[I, 0], res.F[I, 1], color="blue", marker="o", s=80, label="Óptimo Fixed")

# NSGA-II Variable (rojo)
plt.scatter(res_var.F[:, 0], res_var.F[:, 1], facecolors="none", edgecolors="red", marker="o", s=50, label="NSGA-II Variable")
plt.scatter(res_var.F[I_var, 0], res_var.F[I_var, 1], color="red", marker="o", s=80, label="Óptimo Variable")

# Método antiguo de Carles (magenta)
plt.scatter(res_carles.F[:, 0], res_carles.F[:, 1], facecolors="none", edgecolors="magenta", marker="o", s=50, label="NSGA-II Fixed (Carles)")
plt.scatter(res_carles.F[I_carles, 0], res_carles.F[I_carles, 1], color="magenta", marker="o", s=80, label="Óptimo Carles")

plt.xlabel("Coste de Inversión [M€]", fontsize=14, labelpad=20)
plt.ylabel("Coste Técnico [M€]", fontsize=14, labelpad=20)
plt.title("Comparación soluciones NSGA-II\nPotencia Fija (azul) - Potencia Variable (rojo) - Carles (magenta)", fontsize=16, fontweight="bold")

# Ajustar ejes para incluir todos los datos
all_x = np.concatenate([res.F[:, 0], res_var.F[:, 0], res_carles.F[:, 0]])
all_y = np.concatenate([res.F[:, 1], res_var.F[:, 1], res_carles.F[:, 1]])
plt.xlim(all_x.min() - 5, all_x.max() + 5)
plt.ylim(all_y.min() - 50, all_y.max() + 50)

plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()





# GRÁFICA COMPARACIÓN - MÉTODO 1 (NSGA II) VS MÉTODO 2 (Búsqueda Aletoria) - MEJORAR GÁFICA!!!!!!!!!!!!!!!!!!!!!!
  # Comparación gráfica entre NSGA-II y búsqueda aleatoria
import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))

# Búsqueda aleatoria (negro)
plt.scatter(random_check[:, 0], random_check[:, 1],
            facecolors="none", edgecolors="black", marker="o", s=50,
            label="Búsqueda Aleatoria")

# NSGA-II con Potencia Fija (azul)
plt.scatter(res.F[:, 0], res.F[:, 1],
            facecolors="none", edgecolors="blue", marker="o", s=50,
            label="NSGA-II (Fija)")

# NSGA-II con Potencia Variable (rojo)
plt.scatter(res_var.F[:, 0], res_var.F[:, 1],
            facecolors="none", edgecolors="red", marker="o", s=50,
            label="NSGA-II (Variable)")

plt.xlabel("Coste de Inversión [M€]", fontsize=14, labelpad=20)
plt.ylabel("Coste Técnico [M€]", fontsize=14, labelpad=20)

# Título en dos líneas
plt.title("Comparación de soluciones NSGA-II\nPotencia Fija - Potencia Variable - Búsqueda Aleatoria",
          fontsize=16, fontweight="bold")

plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()




plt.figure(figsize=(10,8))

# NSGA-II Fixed (azul)
plt.scatter(res.F[:, 0], res.F[:, 1], facecolors="none", edgecolors="blue", marker="o", s=50, label="NSGA-II Fixed")
plt.scatter(res.F[I, 0], res.F[I, 1], color="blue", marker="o", s=80, label="Óptimo Fixed")

# NSGA-II Variable (rojo)
plt.scatter(res_var.F[:, 0], res_var.F[:, 1], facecolors="none", edgecolors="red", marker="o", s=50, label="NSGA-II Variable")
plt.scatter(res_var.F[I_var, 0], res_var.F[I_var, 1], color="red", marker="o", s=80, label="Óptimo Variable")

# Solución OPF (verde)
plt.scatter(cost_invest_opf, cost_tech_opf, color="green", marker="o", s=100, label="OPF solution")

plt.xlabel("Coste de Inversión [M€]", fontsize=14, labelpad=20)
plt.ylabel("Coste Técnico [M€]", fontsize=14, labelpad=20)
plt.title("Comparación soluciones NSGA II\nPotencia Fija - Potencia Variable vs. OPF", fontsize=16, fontweight="bold")
plt.xlim(150, 205)
plt.ylim(0, 1500)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()





# Gráfics que nos muestran las distintas soluciones mediante optimización - Para validar el uso de NSGA

plt.figure(figsize=(10,8))

# NSGA-II Fixed (azul)
plt.scatter(res.F[:, 0], res.F[:, 1], facecolors="none", edgecolors="blue", marker="o", s=50, label="NSGA-II Fixed")
plt.scatter(res.F[I, 0], res.F[I, 1], color="blue", marker="o", s=80, label="Óptimo Fixed")

# Método antiguo de Carles (magenta)
plt.scatter(res_carles.F[:, 0], res_carles.F[:, 1], facecolors="none", edgecolors="magenta", marker="o", s=50, label="NSGA-II Fixed (Carles)")
plt.scatter(res_carles.F[I_carles, 0], res_carles.F[I_carles, 1], color="magenta", marker="o", s=80, label="Óptimo Carles")

plt.xlabel("Coste de Inversión [M€]", fontsize=14, labelpad=20)
plt.ylabel("Coste Técnico [M€]", fontsize=14, labelpad=20)
plt.title("Comparación soluciones NSGA-II\nPotencia Fija (azul) - Carles (magenta)", fontsize=16, fontweight="bold")
# Ajustar ejes para incluir todos los datos
all_x = np.concatenate([res.F[:, 0], res_var.F[:, 0], res_carles.F[:, 0]])
all_y = np.concatenate([res.F[:, 1], res_var.F[:, 1], res_carles.F[:, 1]])
plt.xlim(all_x.min() - 5, all_x.max() + 5)
plt.ylim(all_y.min() - 50, all_y.max() + 50)

plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()
