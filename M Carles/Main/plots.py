



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









# GRÁFICA 1 - CON COMPENSACIÓN

# Se genera un gráfico de barras para visualizar la distribución de costos en la solución óptima
plt.bar(labels, costs_opf, color='skyblue')
plt.ylabel('Cost [M€]')
plt.title('Breakdown of costs of NSGA-II solution')
plt.xticks(rotation=20, fontsize=18)  # Rotar etiquetas para evitar superposición
# plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Aplicar estilo de Seaborn para mejor visualización
sns.set_style("darkgrid")

# Datos reales que ya tienes en tu código
costs_opf = [c_losses, c_cab, c_gis, c_tr, c_reac, c_ss]  # Costos en M€
labels = ['Power losses', 'Cables', 'Switchgears', 'Transformers', 'Reactors', 'Substation']

# Ajustar el tamaño de la figura
plt.figure(figsize=(10, 6))

# Usar una paleta de colores más estética
colors = sns.color_palette("viridis", len(costs_opf))

# Crear el gráfico de barras con bordes definidos
bars = plt.bar(labels, costs_opf, color=colors, edgecolor='black', linewidth=1.2)

# Etiquetas y títulos mejorados
plt.ylabel('Cost [M€]', fontsize=14, fontweight='bold')
plt.xlabel('Categories', fontsize=14, fontweight='bold')
plt.title('Breakdown of Costs of NSGA-II Solution', fontsize=16, fontweight='bold')

# Ajustar la rotación y el tamaño de las etiquetas del eje X
plt.xticks(rotation=25, fontsize=12)

# Eliminar bordes innecesarios
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Agregar valores encima de las barras para mejor lectura
for bar, cost in zip(bars, costs_opf):
    plt.text(bar.get_x() + bar.get_width() / 2, 
             bar.get_height() + 0.5,  # Ajuste para evitar solapamiento
             f'{cost:.2f} M€', 
             ha='center', fontsize=12, fontweight='bold', color='black')

# Mostrar el gráfico
plt.show()




# GRÁFICA 2 - SIN COMPENSACIÓN

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
# plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Aplicar estilo de Seaborn para mejor visualización
sns.set_style("darkgrid")

# Definir los datos
costs_no = [c_losses, c_cab, c_gis, c_tr, c_reac, c_ss]
labels = ['Power losses', 'Cables', 'Switchgears', 'Transformers', 'Reactors', 'Substation']

# Ajustar el tamaño de la figura
plt.figure(figsize=(10, 6))

# Usar una paleta de colores más estilizada (magma)
colors = sns.color_palette("magma", len(costs_no))

# Crear el gráfico de barras con bordes más definidos
bars = plt.bar(labels, costs_no, color=colors, edgecolor='black', linewidth=1.2)

# Etiquetas y títulos mejorados
plt.ylabel('Cost [M€]', fontsize=14, fontweight='bold')
plt.xlabel('Categories', fontsize=14, fontweight='bold')
plt.title('Breakdown of Costs Without Compensation', fontsize=16, fontweight='bold')

# Ajustar la rotación y el tamaño de las etiquetas del eje X
plt.xticks(rotation=25, fontsize=12)

# Eliminar bordes innecesarios
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Agregar etiquetas de valores encima de las barras para mejor lectura
for bar, cost in zip(bars, costs_no):
    plt.text(bar.get_x() + bar.get_width() / 2, 
             bar.get_height() + 0.5,  # Ajuste para evitar solapamiento
             f'{cost:.2f} M€', 
             ha='center', fontsize=12, fontweight='bold', color='black')

# Mostrar el gráfico
# plt.show()





# GRÁFICA 3 - COMPARACIÓN APILADA

# Se comparan ambas configuraciones con barras apiladas
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
# plt.show()




import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as path_effects  # Para mejorar la visibilidad del texto

# Aplicar un estilo más limpio
sns.set_style("whitegrid")

# Definir el orden de las categorías de abajo a arriba
ordered_labels = ['Cables', 'Substation', 'Power losses', 'Transformers', 'Switchgears', 'Reactors']

# Reordenar los datos según el nuevo orden de etiquetas
costs_opf_reordered = [costs_opf[labels.index(lbl)] for lbl in ordered_labels]
costs_no_reordered = [costs_no[labels.index(lbl)] for lbl in ordered_labels]

# Calcular acumulados para barras apiladas
cumulative_opf = np.cumsum([0] + costs_opf_reordered[:-1])
cumulative_no = np.cumsum([0] + costs_no_reordered[:-1])

# Crear figura con tamaño ajustado
fig, ax = plt.subplots(figsize=(10, 6))

# Usar una paleta de colores más profesional
colors = sns.color_palette("Set2", len(ordered_labels))

# Graficar barras apiladas para NSGA-optimal
for i, cost in enumerate(costs_opf_reordered):
    ax.bar('NSGA-optimal', cost, bottom=cumulative_opf[i], color=colors[i], 
           edgecolor='black', linewidth=1.5, label=ordered_labels[i])

# Graficar barras apiladas para No Compensation
for i, cost in enumerate(costs_no_reordered):
    ax.bar('No Compensation', cost, bottom=cumulative_no[i], color=colors[i], 
           edgecolor='black', linewidth=1.5)

# Agregar etiquetas de valores SOLO Substation
selected_labels = ['Power losses']

for i, (cost, bottom) in enumerate(zip(costs_opf_reordered, cumulative_opf)):
    if ordered_labels[i] in selected_labels:  # Solo mostrar en las barras seleccionadas
        text = ax.text(x=0, y=bottom + cost / 2, s=f'{cost:.2f} M€', ha='center', 
                       fontsize=12, fontweight='bold', color='black')
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])

for i, (cost, bottom) in enumerate(zip(costs_no_reordered, cumulative_no)):
    if ordered_labels[i] in selected_labels:  # Solo mostrar en las barras seleccionadas
        text = ax.text(x=1, y=bottom + cost / 2, s=f'{cost:.2f} M€', ha='center', 
                       fontsize=12, fontweight='bold', color='black')
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])

# Etiquetas y título mejorados
ax.set_ylabel('Cost [M€]', fontsize=16, fontweight='bold')
ax.set_title('Total Cost Comparison', fontsize=18, fontweight='bold')
plt.xticks(rotation=0, fontsize=14)

# Ajustar los límites del eje Y para evitar que se corten las barras
max_value = max(sum(costs_opf_reordered), sum(costs_no_reordered))  # Encontrar la barra más alta
ax.set_ylim(0, max_value * 1.1)  # Dejar un 10% extra de margen arriba

# Crear una única leyenda sin duplicados y con un formato más profesional
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax.legend(unique_labels.values(), unique_labels.keys(), title="Cost Components", 
          bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, frameon=True, fancybox=True)

# Ajustar diseño y mostrar gráfico
plt.tight_layout()
plt.show()
