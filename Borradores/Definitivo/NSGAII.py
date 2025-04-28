# Mmain.py limpio y documentado para el TFG
# ==============================================================
# Este script ejecuta 3 métodos de optimización NSGA-II
# y muestra sus resultados en consola y gráficamente.
# - NSGA-II Carles (potencia fija)
# - NSGA-II Mario (potencia fija)
# - NSGA-II Mario (potencia variable)
# ==============================================================

import numpy as np
import time
import matplotlib.pyplot as plt
from pymoo.optimize import minimize
from pymoo.core.mixed import MixedVariableGA, MixedVariableSampling
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.decomposition.asf import ASF

# ==============================================================
# MÉTODO 1: NSGA-II con potencia fija - CARLES
# ==============================================================
from carles_windopti import MixedVariableProblem as MixedVariableProblemCarles

print("\nEJECUTANDO NSGA-II CARLES (Potencia fija - código original)...")
problem_carles = MixedVariableProblemCarles()

algorithm = MixedVariableGA(
    pop_size=150,
    sampling=MixedVariableSampling(),
    survival=RankAndCrowdingSurvival(crowding_func="pcd"),
    seed=1
)
start_time = time.time()
res_carles = minimize(problem_carles, algorithm, termination=('n_gen', 5), seed=1, verbose=False)
end_time = time.time()
print("Tiempo de ejecución (Carles - P.Fija):", round(end_time - start_time, 2), "s")

weights = np.array([0.5, 0.5])
I_carles = ASF()(res_carles.F, weights).argmin()
print("Solución óptima T.01 (potencia fija):", res_carles.F[I_carles])


# ==============================================================
# MÉTODO 2: NSGA-II con potencia fija - CÓDIGO MODIFICADO (Mario)
# ==============================================================
from windoptiMario import MixedVariableProblem, evaluate_solution

print("\nEJECUTANDO NSGA-II MARIO (Potencia fija)...")
problem_fixed = MixedVariableProblem(tipo_potencia="fija")

algorithm = MixedVariableGA(
    pop_size=150,
    sampling=MixedVariableSampling(),
    survival=RankAndCrowdingSurvival(crowding_func="pcd"),
    seed=1
)
start_time = time.time()
res_fixed = minimize(problem_fixed, algorithm, termination=('n_gen', 5), seed=1, verbose=True)
end_time = time.time()
print("Tiempo de ejecución (Mario - P.Fija):", round(end_time - start_time, 2), "s")

weights = np.array([0.5, 0.5])
I_fixed = ASF()(res_fixed.F, weights).argmin()
print("Solución óptima (potencia fija):", res_fixed.F[I_fixed])

# Evaluar solución óptima con y sin reactores
X_opt_fixed = res_fixed.X[I_fixed]
res_with_react_fixed = evaluate_solution(problem_fixed, X_opt_fixed.copy(), disable_reactors=False)
res_without_react_fixed = evaluate_solution(problem_fixed, X_opt_fixed.copy(), disable_reactors=True)

#print(">>> Costes óptimos con shunt reactors (potencia fija):", res_with_react_fixed)
#print(">>> Costes óptimos sin shunt reactors (potencia fija):", res_without_react_fixed)


# ==============================================================
# MÉTODO 3: NSGA-II con potencia variable media - CÓDIGO MODIFICADO (Mario)
# ==============================================================
print("\nEJECUTANDO NSGA-II MARIO (Potencia variable media)...")
problem_var = MixedVariableProblem(tipo_potencia="variable")

algorithm = MixedVariableGA(
    pop_size=150,
    sampling=MixedVariableSampling(),
    survival=RankAndCrowdingSurvival(crowding_func="pcd"),
    seed=1
)
start_time = time.time()
res_var = minimize(problem_var, algorithm, termination=('n_gen', 5), seed=1, verbose=False)
end_time = time.time()
print("Tiempo de ejecución (Mario - P.Variable media):", round(end_time - start_time, 2), "s")

I_var = ASF()(res_var.F, weights).argmin()
print("Solución óptima (potencia variable media):", res_var.F[I_var])

X_opt_var = res_var.X[I_var]
res_with_react_var = evaluate_solution(problem_var, X_opt_var.copy(), disable_reactors=False)
res_without_react_var = evaluate_solution(problem_var, X_opt_var.copy(), disable_reactors=True)

#print(">>> Costes óptimos con shunt reactors (potencia variable media):", res_with_react_var)
#print(">>> Costes óptimos sin shunt reactors (potencia variable media):", res_without_react_var)

# ==============================================================
# MÉTODO 4: NSGA-II con potencia variable - CÓDIGO MODIFICADO (Mario)
# ==============================================================
print("\nEJECUTANDO NSGA-II MARIO (Potencia variable extendida)...")
problem_ext = MixedVariableProblem(tipo_potencia="variable_extendida")

algorithm = MixedVariableGA(
    pop_size=150,
    sampling=MixedVariableSampling(),
    survival=RankAndCrowdingSurvival(crowding_func="pcd"),
    seed=1
)
start_time = time.time()
res_ext = minimize(problem_ext, algorithm, termination=('n_gen', 5), seed=1, verbose=False)
end_time = time.time()
print("Tiempo de ejecución (Mario - P.Variable):", round(end_time - start_time, 2), "s")

I_ext = ASF()(res_ext.F, weights).argmin()
print("Solución óptima (potencia variable extendida):", res_ext.F[I_ext])

X_opt_ext = res_ext.X[I_ext]
res_with_react_ext = evaluate_solution(problem_ext, X_opt_ext.copy(), disable_reactors=False)
res_without_react_ext = evaluate_solution(problem_ext, X_opt_ext.copy(), disable_reactors=True)

#print(">>> Costes óptimos con shunt reactors (potencia variable extendida):", res_with_react_ext)
#print(">>> Costes óptimos sin shunt reactors (potencia variable extendida):", res_without_react_ext)




import pandas as pd

# ----------- RECOGER RESULTADOS -----------
labels = [
    "vol_penal", "curr_penal", "losses", "react_penal", "coste_tec",
    "c_cables", "c_gis", "c_trafo", "c_reactores", "c_subest", "coste_inv"
]

resultados = {
    "Mario - Potencia fija (SI compensación)": res_with_react_fixed,
    "Mario - Potencia fija (NO compensación)": res_without_react_fixed,
    "Mario - Variable media (SI compensación)": res_with_react_var,
    "Mario - Variable media (NO compensación)": res_without_react_var,
    "Mario - Variable completa (SI compensación)": res_with_react_ext,
    "Mario - Variable completa (NO compensación)": res_without_react_ext,
}

# ----------- CONSTRUIR TABLA -----------
df_resultados = pd.DataFrame.from_dict(resultados, orient='index')[labels]
df_resultados.index.name = "Método"
df_resultados.columns = [
    "Penalización voltaje (M€)", "Penalización corriente (M€)", "Pérdidas (M€)",
    "Penalización reactiva (M€)", "Coste técnico (M€)",
    "Cables (M€)", "Switchgears (M€)", "Transformadores (M€)",
    "Reactores (M€)", "Subestación (M€)", "Coste inversión (M€)"
]

# ----------- MOSTRAR Y GUARDAR -----------
print("\n=========== COMPARATIVA FINAL DE COSTES ===========")
print(df_resultados.round(2))

# (Opcional) Exportar a CSV
df_resultados.to_csv("comparativa_costes_Mario.csv", sep=";", index=True)

# ==============================================================
# GUARDAR LOS DATOS
# ==============================================================
import json

# Crear diccionario con los resultados
resultados = {
    "Potencia Fija": {
        "con": res_with_react_fixed,
        "sin": res_without_react_fixed
    },
    "Potencia Variable Media": {
        "con": res_with_react_var,
        "sin": res_without_react_var
    },
    "Potencia Variable": {
        "con": res_with_react_ext,
        "sin": res_without_react_ext
    }
}

# Convertir arrays NumPy a listas (para poder serializar)
for metodo in resultados:
    for caso in resultados[metodo]:
        resultados[metodo][caso] = {k: float(v) for k, v in resultados[metodo][caso].items()}

# Guardar resultados en JSON
with open("resultados_optimos.json", "w") as f:
    json.dump(resultados, f, indent=4)


# ==============================================================
# GRÁFICAS COMPARATIVAS
# ==============================================================
# Asegurarse de que todos los frentes de Pareto sean 2D para graficar correctamente
if res_carles.F.ndim == 1:
    res_carles.F = np.array([res_carles.F])
if res_fixed.F.ndim == 1:
    res_fixed.F = np.array([res_fixed.F])
if res_var.F.ndim == 1:
    res_var.F = np.array([res_var.F])
if res_ext.F.ndim == 1:
    res_ext.F = np.array([res_ext.F])

from pymoo.visualization.scatter import Scatter

# GRÁFICA INDIVIDUAL – T.01
plt.figure(figsize=(6,10))
plt.scatter(res_carles.F[:, 0], res_carles.F[:, 1], facecolors="none", edgecolors="black", marker="o", s=50, label="NSGA-II (Potencia Fija) T.01")
plt.scatter(res_carles.F[I_carles, 0], res_carles.F[I_carles, 1], color="red", marker="o", s=80, label="Óptimo")
plt.xlabel("Coste de Inversión [M€]", fontsize=14, labelpad=10)
plt.ylabel("Coste Técnico [M€]", fontsize=14, labelpad=10)
plt.xlim(180, 205)
plt.ylim(0, 1400)
plt.title("NSGA-II (Potencia Fija) T.01", fontsize=16, fontweight="bold")
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# GRÁFICA INDIVIDUAL – Mario Potencia Fija
plt.figure(figsize=(6,10))
plt.scatter(res_fixed.F[:, 0], res_fixed.F[:, 1], facecolors="none", edgecolors="black", marker="o", s=50, label="NSGA-II (Potencia Fija)")
plt.scatter(res_fixed.F[I_fixed, 0], res_fixed.F[I_fixed, 1], color="red", marker="o", s=80, label="Óptimo")
plt.xlabel("Coste de Inversión [M€]", fontsize=14, labelpad=10)
plt.ylabel("Coste Técnico [M€]", fontsize=14, labelpad=10)
plt.xlim(180, 205)
plt.ylim(0, 1500)
plt.title("NSGA-II (Potencia Fija)", fontsize=16, fontweight="bold")
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# GRÁFICA INDIVIDUAL – Mario Potencia Variable media
plt.figure(figsize=(6,10))
plt.scatter(res_var.F[:, 0], res_var.F[:, 1], facecolors="none", edgecolors="black", marker="o", s=50, label="NSGA-II Variable media")
plt.scatter(res_var.F[I_var, 0], res_var.F[I_var, 1], color="red", marker="o", s=80, label="Óptimo Variable media")
plt.xlabel("Coste de Inversión [M€]", fontsize=14, labelpad=10)
plt.ylabel("Coste Técnico [M€]", fontsize=14, labelpad=10)
plt.xlim(150, 185)
plt.ylim(0, 2000)
plt.title("NSGA-II (Potencia Variable media)", fontsize=16, fontweight="bold")
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# GRÁFICA INDIVIDUAL – Mario Potencia Variable 
plt.figure(figsize=(8,8))
plt.scatter(res_ext.F[:, 0], res_ext.F[:, 1], facecolors="none", edgecolors="black", marker="o", s=50, label="NSGA-II Variable")
plt.scatter(res_ext.F[I_ext, 0], res_ext.F[I_ext, 1], color="red", marker="o", s=80, label="Óptimo Variable")
plt.xlabel("Coste de Inversión [M€]", fontsize=14, labelpad=10)
plt.ylabel("Coste Técnico [M€]", fontsize=14, labelpad=10)
plt.xlim(150, 205)
plt.ylim(0, 1600)
plt.title("NSGA-II (Potencia Variable)", fontsize=16, fontweight="bold")
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

#GRÁFICA COMPARATIVA T.01 VS MARIO FIJA
plt.figure(figsize=(6,10))
plt.scatter(res_fixed.F[:, 0], res_fixed.F[:, 1], facecolors="none", edgecolors="blue", marker="o", s=50, label="NSGA-II Fija")
plt.scatter(res_fixed.F[I_fixed, 0], res_fixed.F[I_fixed, 1], color="blue", marker="o", s=80, label="Óptimo Fixed")
plt.scatter(res_carles.F[:, 0], res_carles.F[:, 1], facecolors="none", edgecolors="magenta", marker="o", s=50, label="NSGA-II Fija (T.01)")
plt.scatter(res_carles.F[I_carles, 0], res_carles.F[I_carles, 1], color="magenta", marker="o", s=80, label="Óptimo T.01")
plt.xlabel("Coste de Inversión [M€]", fontsize=14, labelpad=20)
plt.ylabel("Coste Técnico [M€]", fontsize=14, labelpad=20)
plt.title("Comparación soluciones NSGA-II\nPotencia Fija (azul) - Potencia Fija T.01 (magenta)", fontsize=16, fontweight="bold")
plt.xlim(180, 205)
plt.ylim(0, 1600)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# GRÁFICA COMPARATIVA T.01 VS MARIO FIJA VS MARIO VARIABLE-MEDIA
plt.figure(figsize=(6,12))
plt.scatter(res_fixed.F[:, 0], res_fixed.F[:, 1], facecolors="none", edgecolors="blue", marker="o", s=50, label="NSGA-II Fija")
plt.scatter(res_fixed.F[I_fixed, 0], res_fixed.F[I_fixed, 1], color="blue", marker="o", s=80, label="Óptimo Fixed")
plt.scatter(res_carles.F[:, 0], res_carles.F[:, 1], facecolors="none", edgecolors="magenta", marker="o", s=50, label="NSGA-II Fixed (T.01)")
plt.scatter(res_carles.F[I_carles, 0], res_carles.F[I_carles, 1], color="magenta", marker="o", s=80, label="Óptimo T.01")
plt.scatter(res_var.F[:, 0], res_var.F[:, 1], facecolors="none", edgecolors="red", marker="o", s=50, label="NSGA-II Variable-Media")
plt.scatter(res_var.F[I_var, 0], res_var.F[I_var, 1], color="red", marker="o", s=80, label="Óptimo Variable-Media")
plt.xlabel("Coste de Inversión [M€]", fontsize=14, labelpad=20)
plt.ylabel("Coste Técnico [M€]", fontsize=14, labelpad=20)
plt.title("Comparación soluciones NSGA-II\nPotencia Fija (azul) - Potencia Variable-Media (rojo) - T.01 (magenta)", fontsize=16, fontweight="bold")
plt.xlim(150, 205)
plt.ylim(0, 1600)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# GRÁFICA COMPARATIVA MARIO FIJA VS MARIO VARIABLE-MEDIA
#plt.figure(figsize=(10,8))
#plt.scatter(res_fixed.F[:, 0], res_fixed.F[:, 1], facecolors="none", edgecolors="blue", marker="o", s=50, label="NSGA-II Fixed")
#plt.scatter(res_fixed.F[I_fixed, 0], res_fixed.F[I_fixed, 1], color="blue", marker="o", s=80, label="Óptimo Fixed")
#plt.scatter(res_var.F[:, 0], res_var.F[:, 1], facecolors="none", edgecolors="red", marker="o", s=50, label="NSGA-II Variable")
#plt.scatter(res_var.F[I_var, 0], res_var.F[I_var, 1], color="red", marker="o", s=80, label="Óptimo Variable")
#plt.xlabel("Coste de Inversión [M€]", fontsize=14, labelpad=20)
#plt.ylabel("Coste Técnico [M€]", fontsize=14, labelpad=20)
#plt.title("Comparación soluciones NSGA-II\nPotencia Fija (azul) - Potencia Variable (rojo)", fontsize=16, fontweight="bold")
#plt.xlim(150, 205)
#plt.ylim(0, 1300)
#plt.legend(fontsize=12)
#plt.grid(True, linestyle="--", alpha=0.7)
#plt.show()

# GRÁFICA COMPARATIVA MARIO FIJA VS MARIO VARIABLE-MEDIA VS MARIO VARIABLE
plt.figure(figsize=(11,13))
plt.scatter(res_fixed.F[:, 0], res_fixed.F[:, 1], facecolors="none", edgecolors="blue", marker="o", s=50, label="NSGA-II Fixed")
plt.scatter(res_fixed.F[I_fixed, 0], res_fixed.F[I_fixed, 1], color="blue", marker="o", s=80, label="Óptimo Fija")
plt.scatter(res_var.F[:, 0], res_var.F[:, 1], edgecolors="red", facecolors='none', label="NSGA-II Variable-Media")
plt.scatter(res_var.F[I_var, 0], res_var.F[I_var, 1], color="red", marker="o", s=80, label="Óptimo Variable-Media")
plt.scatter(res_ext.F[:, 0], res_ext.F[:, 1], edgecolors="orange", facecolors='none', label="NSGA-II Variable")
plt.scatter(res_ext.F[I_ext, 0], res_ext.F[I_ext, 1], color="orange", marker="o", s=80, label="Óptimo Variable")
plt.xlabel("Coste de Inversión [M€]", fontsize=14, labelpad=20)
plt.ylabel("Coste Técnico [M€]", fontsize=14, labelpad=20)
plt.title("Comparación soluciones NSGA-II\nPotencia Fija (azul) - Potencia Variable-Media (rojo) - -Potencia Variable (naranja)", fontsize=16, fontweight="bold")
plt.xlim(150, 215)
plt.ylim(0, 1600)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

#GRÁFICA COMPARATIVA T.01 VS MARIO FIJA VS MARIO VARIABLE-MEDIA VS MARIO VARIABLE
#plt.figure(figsize=(10,8))
#plt.scatter(res_carles.F[:, 0], res_carles.F[:, 1], facecolors="none", edgecolors="magenta", marker="o", s=50, label="NSGA-II Fixed (Carles)")
#plt.scatter(res_carles.F[I_carles, 0], res_carles.F[I_carles, 1], color="magenta", marker="o", s=80, label="Óptimo Carles")
#plt.scatter(res_fixed.F[:, 0], res_fixed.F[:, 1], edgecolors="blue", facecolors='none', label="Potencia Fija")
#plt.scatter(res_fixed.F[I_fixed, 0], res_fixed.F[I_fixed, 1], color="blue", marker="o", s=80, label="Óptimo Fija")
#plt.scatter(res_var.F[:, 0], res_var.F[:, 1], edgecolors="orange", facecolors='none', label="P.Variable Media")
#plt.scatter(res_var.F[I_var, 0], res_var.F[I_var, 1], color="orange", marker="o", s=80, label="Óptimo Variable media")
#plt.scatter(res_ext.F[:, 0], res_ext.F[:, 1], edgecolors="red", facecolors='none', label="P.Variable Completa")
#plt.scatter(res_ext.F[I_ext, 0], res_ext.F[I_ext, 1], color="red", marker="o", s=80, label="Óptimo Variable Completa")
#plt.xlabel("Coste de Inversión [M€]")
#plt.ylabel("Coste Técnico [M€]")
#plt.legend()
#plt.title("Comparación modos de potencia")
#plt.grid(True, linestyle="--", alpha=0.7)
#plt.show()