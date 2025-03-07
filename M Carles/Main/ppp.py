



import numpy as np                                                          # Importa la biblioteca NumPy para cálculos numéricos
import matplotlib as mpl                                                    # Biblioteca para configuraciones avanzadas de gráficos
import matplotlib.pyplot as plt                                             # Biblioteca para graficar
import time                                                                 # Biblioteca para medir tiempos de ejecución

# Llamada a nuestros programas
from windopti import MixedVariableProblem                                   # Importa la clase que define el problema de optimización
from costac_2 import costac_2                                              # Importa la función de costos del sistema

# pymoo para trabajar con variables mixtas
from pymoo.algorithms.moo.nsga2 import NSGA2                                # Importa el algoritmo NSGA-II para optimización multiobjetivo
from pymoo.problems import get_problem                                      # Importa función para obtener problemas de optimización predefinidos
from pymoo.optimize import minimize                                         # Función para minimizar un problema de optimización
from pymoo.visualization.scatter import Scatter                             # Herramienta para visualizar resultados
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.algorithms.moo.nsga2 import RankAndCrowding
from pymoo.constraints.as_penalty import ConstraintsAsPenalty
from pymoo.decomposition.asf import ASF                                    # Función de escalarización para selección de solución óptima
from pymoo.core.evaluator import Evaluator                                 # Evaluador de soluciones en pymoo
from pymoo.core.individual import Individual                               # Representa una solución en la población
from pymoo.operators.sampling.rnd import FloatRandomSampling               # Muestreo aleatorio de valores flotantes
from pymoo.core.mixed import MixedVariableSampling                         # Muestreo de variables mixtas en pymoo



# Se define el problema de optimización
problem = MixedVariableProblem()

# Se define la función de costos a usar
ff = costac_2
p_owf = 5  # Potencia del parque eólico

# Se define una solución óptima encontrada por OPF
x_opf = np.array([3, 2, 1, 1, 0, 1, 0, 0.519, 0.953, 0.0, 0.737, 0.0, 509.72e6])

# Se extraen los valores de la solución óptima (Mmain)
vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr = x_opf

# Se calcula el costo de inversión y técnico usando la función de costos (costac)
cost_invest_opf, cost_tech_opf, cost_fullopf = ff(vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr, p_owf)

# Se extraen costos específicos del resultado de costos (cosatc)
c_vol, c_curr, c_losses, c_react, cost_tech, c_cab, c_gis, c_tr, c_reac, cost_invest,c_volover, c_volunder, c_ss, average_v = cost_fullopf

# Se organizan los costos para graficarlos
costs_opf= [c_losses, c_cab, c_gis, c_tr, c_reac, c_ss]
labels = ['Power losses', 'Cables', 'Switchgears', 'Transformers', 'Reactors', 'Substation']



# Se genera un gráfico de barras para visualizar la distribución de costos en la solución óptima
plt.bar(labels, costs_opf, color='skyblue')
plt.ylabel('Cost [M€]')
plt.title('Breakdown of costs of NSGA-II solution')
plt.xticks(rotation=20, fontsize=18)  # Rotar etiquetas para evitar superposición
#plt.show()



# Se define otra solución sin compensación reactiva
x_nosh = np.array([3, 2, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 509.72e6])
vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr = x_nosh

# Se calcula el costo de inversión y técnico sin compensación
cost_invest_no, cost_tech_no, cost_full_no = ff(vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr, p_owf)
c_vol, c_curr, c_losses, c_react, cost_tech, c_cab, c_gis, c_tr, c_reac, cost_invest,c_volover, c_volunder, c_ss, average_v = cost_full_no

costs_no= [c_losses, c_cab, c_gis, c_tr, c_reac, c_ss]
labels = ['Power losses', 'Cables', 'Switchgears', 'Transformers', 'Reactors', 'Substation']

# Se grafica el desglose de costos sin compensación
plt.bar(labels, costs_no, color='orange')  # Se superpone sobre el gráfico anterior
plt.ylabel('Cost [M€]')
plt.title('Breakdown of costs without compensation')
plt.xticks(rotation=20, fontsize= 18)  # Rotar etiquetas para evitar superposición
plt.show()

# Se comparan ambas configuraciones con barras apiladas
labels = ['Power losses', 'Cables', 'Switchgears', 'Transformers', 'Reactors', 'Substation']
cumulative_opf = np.cumsum([0] + costs_opf[:-1])
cumulative_no = np.cumsum([0] + costs_no[:-1])
fig, ax = plt.subplots()

# Se grafican los costos con compensación
for i, cost in enumerate(costs_opf):
    ax.bar('NSGA-optimal', cost, bottom=cumulative_opf[i], color=plt.cm.Paired(i), label=labels[i])

# Se grafican los costos sin compensación
for i, cost in enumerate(costs_no):
    ax.bar('No Compensation', cost, bottom=cumulative_no[i], color=plt.cm.Paired(i), label=labels[i] if i == 0 else "")

# Etiquetas y título del gráfico
ax.set_ylabel('Cost [M€]')
ax.set_title('Total Cost Comparison')
plt.xticks(rotation=0)  # Rotación de etiquetas para mejorar legibilidad

# Se crea la leyenda evitando duplicados
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
legend = plt.legend(unique_labels.values(), unique_labels.keys(), title="Cost Components", bbox_to_anchor=(1.05, 1), loc='upper left')

# Se muestra el gráfico final
plt.tight_layout()
plt.show()