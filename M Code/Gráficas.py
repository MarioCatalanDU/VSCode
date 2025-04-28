# Gráficas.py - Comparaciones visuales finales de escenarios

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as path_effects

# Para re-ejecutar NSGA-II y evaluar pérdidas
from Optimizador import MixedVariableProblem, evaluate_solution
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableGA, MixedVariableSampling
from pymoo.decomposition.asf import ASF

# Crear carpeta para guardar gráficas
os.makedirs("figures", exist_ok=True)

# ======= CARGAR RESULTADOS DESDE JSON =======
with open("resultados_optimos.json", "r") as f:
    datos_costes = json.load(f)

# ——— FILTRAR SOLO COMPONENTES REALES (sin penalizaciones) ———
componentes_reales = ["losses", "c_cables", "c_gis", "c_trafo", "c_reactores", "c_subest"]
for metodo in datos_costes:
    for caso in ("con", "sin"):
        datos_costes[metodo][caso] = {
            k: v for k, v in datos_costes[metodo][caso].items()
            if k in componentes_reales
        }

metodos = list(datos_costes.keys())

# ======= GRÁFICA 1: Costes totales con y sin compensación (sin penalizaciones) =======
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(metodos))
width = 0.35

costes_con = [sum(datos_costes[m]["con"].values()) for m in metodos]
costes_sin = [sum(datos_costes[m]["sin"].values()) for m in metodos]

bars1 = ax.bar(x - width/2, costes_con, width, label='Con compensación', color='#1f77b4', edgecolor='black')
bars2 = ax.bar(x + width/2, costes_sin, width, label='Sin compensación', color='#ff7f0e', edgecolor='black')

for bars in (bars1, bars2):
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval * 1.01, f"{yval:.1f}",
                ha='center', fontsize=11, fontweight='bold', color='black',
                path_effects=[path_effects.Stroke(linewidth=1.5, foreground='white'), path_effects.Normal()])

ax.set_ylabel("Coste total [M€]", fontsize=13)
ax.set_title("Coste total con y sin compensación reactiva", fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metodos, rotation=0)
ax.legend()
plt.tight_layout()
plt.savefig("figures/coste_total_comparativa.png")
plt.show()

# ======= GRÁFICAS 2 a 4: Desglose de costes por método (con vs sin) =======
for metodo in metodos:
    fig, ax = plt.subplots(figsize=(10, 6))
    etiquetas = ["Pérdidas", "Cables", "Switchgears", "Transformadores", "Reactores", "Subestación"]
    componentes = componentes_reales
    con = [datos_costes[metodo]["con"].get(k, 0.0) for k in componentes]
    sin = [datos_costes[metodo]["sin"].get(k, 0.0) for k in componentes]

    x = np.arange(len(componentes))
    width = 0.35

    bars_con = ax.bar(x - width/2, con, width, label="Con comp.", color='#1f77b4', edgecolor='black')
    bars_sin = ax.bar(x + width/2, sin, width, label="Sin comp.", color='#ff7f0e', edgecolor='black')

    for bars in (bars_con, bars_sin):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height * 1.01, f"{height:.1f}",
                    ha='center', fontsize=11, fontweight='bold', color='black',
                    path_effects=[path_effects.Stroke(linewidth=1.5, foreground='white'), path_effects.Normal()])

    ax.set_ylabel("Coste [M€]", fontsize=13)
    ax.set_title(f"Desglose de costes - {metodo}", fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, rotation=25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"figures/desglose_costes_{metodo.replace(' ', '_').lower()}.png")
    plt.show()

# ======= GRÁFICA NUEVA 1: Pérdidas vs Potencia Inyectada =======
# 1) Re-ejecutar NSGA-II para obtener el diseño óptimo de potencia fija
peso = np.array([0.5, 0.5])
problem_fixed = MixedVariableProblem(tipo_potencia="fija")
alg_fixed = MixedVariableGA(
    pop_size=150,
    sampling=MixedVariableSampling(),
    survival=RankAndCrowdingSurvival(crowding_func="pcd"),
    seed=1
)
res_fixed = minimize(problem_fixed, alg_fixed, termination=('n_gen', 5), seed=1, verbose=False)
I_fixed = ASF()(res_fixed.F, peso).argmin()
X_opt_fixed = res_fixed.X[I_fixed]

# 2) Barrer inyecciones de potencia en p.u.
p_range = np.linspace(2.0, 5.0, 50)
losses_no_comp = []
losses_opt_comp = []
for p in p_range:
    # redefinir escenario a potencia p
    problem_fixed.scenarios = [(p, 1.0)]
    losses_no_comp.append(evaluate_solution(problem_fixed, X_opt_fixed.copy(), disable_reactors=True)["losses"])
    losses_opt_comp.append(evaluate_solution(problem_fixed, X_opt_fixed.copy(), disable_reactors=False)["losses"])

# 3) Graficar
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(p_range, losses_no_comp, label='Sin compensación', linestyle='--', linewidth=2)
ax.plot(p_range, losses_opt_comp, label='Con compensación', linestyle='-', linewidth=2)
# Destacar mínimo con compensación
i_min = np.argmin(losses_opt_comp)
p_min, loss_min = p_range[i_min], losses_opt_comp[i_min]
ax.scatter(p_min, loss_min, color='red', zorder=5)
ax.annotate(f"Mínimo ({p_min:.2f} p.u., {loss_min:.1f} M€)",
            xy=(p_min, loss_min), xytext=(p_min + 0.2, loss_min + 10),
            arrowprops=dict(arrowstyle='->', lw=1.5), fontsize=12)
ax.set_xlabel("Potencia generada [p.u.]", fontsize=14)
ax.set_ylabel("Costes anuales [M€]", fontsize=14)
ax.set_title("Costes potencia perdida vs Potencia generada (500 MW OWPP)", fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("figures/losses_vs_injection.png")
plt.show()

# ======= GRÁFICA NUEVA 2: Comparativa de pérdidas entre escenarios =======
#fig, ax = plt.subplots(figsize=(8, 6))
#losses_vals = [datos_costes[m]["con"]["losses"] for m in metodos]
#bars = ax.bar(metodos, losses_vals, color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black')
#for bar in bars:
#    h = bar.get_height()
#    ax.text(bar.get_x() + bar.get_width()/2, h * 1.01, f"{h:.1f}",
#            ha='center', fontsize=11, fontweight='bold', color='black')
#ax.set_ylabel("Annual losses cost [M€]", fontsize=13)
#ax.set_title("Comparativa de pérdidas anuales por escenario", fontsize=15, fontweight='bold')
#ax.set_xticklabels(metodos, rotation=15)
#ax.grid(axis='y', linestyle='--', alpha=0.7)
#plt.tight_layout()
#plt.savefig("figures/losses_comparison_scenarios.png")
#plt.show()
