# DIAGRAMAS DE BARRAS de COSTES - GRÁFICAS Costes para windopti

 # COMPARACIÓN costes : Compensación vs No Compensación - Para validar con Costes el uso de NSGA 
 # Usando la solución extraida de NSGA (CON Compensación) graficamos sus costes VS la misma solución pero (SIN Compensación)





# 1. IMPORTAR HERRAMIENTAS

import numpy as np                                                          # Importa la biblioteca NumPy para cálculos numéricos
import matplotlib as mpl                                                    # Biblioteca para configuraciones avanzadas de gráficos
import matplotlib.pyplot as plt                                             # Biblioteca para graficar
import time                                                                 # Biblioteca para medir tiempos de ejecución

import seaborn as sns                                                       # Para mejorar gráficas
import matplotlib.patheffects as path_effects                               # Para mejorar la visibilidad del texto

# Importación de Nuestras funciones de optimización
from windopti import MixedVariableProblem                                   # Importa la clase que define el problema de optimización
from costac_2 import costac_2                                               # Importa la función de costos del sistema

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





# 2. DEFINICIÓN DE VALORES - Extraidos de windopti y costac_2

problem = MixedVariableProblem()                                           # Se define el problema de optimización
ff = costac_2                                                              # Se define la función de costos a usar
p_owf = 5                                                                  # Potencia del parque eólico
 




# 3. SOLUCIÓN NSGA II - CON COMPENSACIÓN

# Aplicamos la solución óptima encontrada por Mmain
x_opf = np.array([
    1,                  # react1_bi : variables binarias (0 o 1), que indican si un reactor está activado (1) o desactivado (0)
    1,                  # react2_bi    
    0,                  # react3_bi
    1,                  # react4_bi
    0,                  # react5_bi
    3,                  # vol_level
    2,                  # n_cables: Número de cables en paralelo. Solo puede tomar valores enteros entre 2 y 3
    581.72e6,           # S_rtr: Potencia nominal del transformador, que puede variar entre 300 MW y 900 MW   
    0.944,              # react1_val : Tamaños de los reactores, definidos como valores continuos entre 0.0 y 1.0 
    0.845,              # react2_val
    0.0,                # react3_val
    0.892,              # react4_val
    0.0,                # react5_val
    ])
# Se extraen los valores de la solución óptima 
react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, vol, n_cables,  S_rtr, react1_val, react2_val, react3_val,react4_val, react5_val = x_opf
# Se calcula el costo de inversión y técnico usando la función de costos 
cost_invest_opf, cost_tech_opf, cost_fullopf = ff(vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr, p_owf)
# Se extraen costos específicos del resultado de costos 
c_vol, c_curr, c_losses, c_react, cost_tech, c_cab, c_gis, c_tr, c_reac, cost_invest,c_volover, c_volunder, c_ss, average_v = cost_fullopf
# Se organizan los costos para graficarlos
costs_opf= [c_losses, c_cab, c_gis, c_tr, c_reac, c_ss]
labels = ['Power losses', 'Cables', 'Switchgears', 'Transformers', 'Reactors', 'Substation']


# GRÁFICA 1 - CON COMPENSACIÓN

#####

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
plt.title('Breakdown of Costs of NSGA-II (Fixed Power) Solution', fontsize=16, fontweight='bold')

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





# 4.  SOLUCIÓN NSGA II - SIN COMPENSACIÓN
  
# Se define la misma solución pero sin compensación reactiva - Todos los Reactores = 0 - Como si estuvieran apagados
x_nosh = np.array([
    3,                 # vol_level
    2,                 # n_cables: Número de cables en paralelo. Solo puede tomar valores enteros entre 2 y 3
    0,                 # 0 - descativado
    0,                 # 0 - descativado
    0,                 # 0 - descativado
    0,                 # 0 - descativado
    0,                 # 0 - descativado
    0.0,               # 0 - descativado
    0.0,               # 0 - descativado
    0.0,               # 0 - descativado
    0.0,               # 0 - descativado
    0.0,               # 0 - descativado
    509.72e6           # S_rtr: Potencia nominal del transformador, que puede variar entre 300 MW y 900 MW 
    ])
# Se extraen los valores de la solución
vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr = x_nosh
# Se calcula el costo de inversión y técnico sin compensación
cost_invest_no, cost_tech_no, cost_full_no = ff(vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr, p_owf)
# Se extraen costos específicos del resultado
c_vol, c_curr, c_losses, c_react, cost_tech, c_cab, c_gis, c_tr, c_reac, cost_invest,c_volover, c_volunder, c_ss, average_v = cost_full_no
# Se organizan los costos para graficarlos
costs_no= [c_losses, c_cab, c_gis, c_tr, c_reac, c_ss]
labels = ['Power losses', 'Cables', 'Switchgears', 'Transformers', 'Reactors', 'Substation']


# GRÁFICA 2 - SIN COMPENSACIÓN

####

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
plt.title('Breakdown of Costs Without Compensation (Fixed Power)', fontsize=16, fontweight='bold')

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


####

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
ax.set_title('Total Cost Comparison (Fixed Power)', fontsize=18, fontweight='bold')
plt.xticks(rotation=0, fontsize=14)

# Ajustar los límites del eje Y para evitar que se corten las barras
max_value = max(sum(costs_opf_reordered), sum(costs_no_reordered))  # Encontrar la barra más alta
ax.set_ylim(0, max_value * 1.1)  # Dejar un 10% extra de margen arriba

# Crear una única leyenda sin duplicados y con un formato más profesional
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax.legend(unique_labels.values(), unique_labels.keys(), title="Cost Components (Fixed Power)", 
          bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, frameon=True, fancybox=True)

# Ajustar diseño y mostrar gráfico
plt.tight_layout()
plt.show()



# plots_var.py - Gráficos utilizando resultados de optimización con potencia variable

# 1. IMPORTAR HERRAMIENTAS
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

import seaborn as sns
import matplotlib.patheffects as path_effects

# Importar funciones de optimización de la versión de potencia variable
from var_windopti import MixedVariableProblem  # Se asume que la interfaz es similar
from costac_2 import costac_2

# 2. DEFINICIÓN DE VALORES - Extraídos de la optimización con potencia variable
# Para el caso CON COMPENSACIÓN:
# Nota: El arreglo x_opf_var contiene 14 elementos:
# [react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, vol_level, n_cables, S_rtr, 
#  react1_val, react2_val, react3_val, react4_val, react5_val, p_var]
x_opf_var = np.array([
    1,      # react1_bi: reactor activado
    1,      # react2_bi
    0,      # react3_bi
    1,      # react4_bi
    0,      # react5_bi
    3,      # vol_level
    2,      # n_cables
    581.72e6,   # S_rtr: Potencia nominal del transformador
    0.944,  # react1_val
    0.845,  # react2_val
    0.0,    # react3_val
    0.892,  # react4_val
    0.0,    # react5_val
    4.3     # p_var: Potencia variable calculada (valor de ejemplo)
])

# Para el caso SIN COMPENSACIÓN:
# Se asume que la solución se obtiene con todos los reactores apagados y se incluye el p_var calculado.
# En este caso, se define un arreglo de 14 elementos:
# [vol_level, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi,
#  react1_val, react2_val, react3_val, react4_val, react5_val, S_rtr, p_var]
x_nosh_var = np.array([
    3,          # vol_level
    2,          # n_cables
    0,          # react1_bi (apagado)
    0,          # react2_bi
    0,          # react3_bi
    0,          # react4_bi
    0,          # react5_bi
    0.0,        # react1_val
    0.0,        # react2_val
    0.0,        # react3_val
    0.0,        # react4_val
    0.0,        # react5_val
    509.72e6,   # S_rtr: Potencia nominal del transformador
    4.1         # p_var: Potencia variable calculada (valor de ejemplo)
])

# Extraer valores para el caso CON COMPENSACIÓN
react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, vol, n_cables, S_rtr, \
react1_val, react2_val, react3_val, react4_val, react5_val, p_var = x_opf_var

# Calcular los costos utilizando la función de costos (se utiliza la potencia variable p_var)
cost_invest_opf, cost_tech_opf, cost_fullopf = costac_2(vol, n_cables, 
    react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, 
    react1_val, react2_val, react3_val, react4_val, react5_val, 
    S_rtr, p_var)

# Extraer costos específicos del resultado de costos
c_vol, c_curr, c_losses, c_react, cost_tech, c_cab, c_gis, c_tr, \
c_reac, cost_invest, c_volover, c_volunder, c_ss, average_v = cost_fullopf

# Organizar los costos para graficarlos
costs_opf = [c_losses, c_cab, c_gis, c_tr, c_reac, c_ss]
labels = ['Power losses', 'Cables', 'Switchgears', 'Transformers', 'Reactors', 'Substation']

# GRÁFICA 1 - CON COMPENSACIÓN
sns.set_style("darkgrid")
plt.figure(figsize=(10, 6))
colors = sns.color_palette("viridis", len(costs_opf))
bars = plt.bar(labels, costs_opf, color=colors, edgecolor='black', linewidth=1.2)
plt.ylabel('Cost [M€]', fontsize=14, fontweight='bold')
plt.xlabel('Categories', fontsize=14, fontweight='bold')
plt.title('Breakdown of Costs of NSGA-II (Variable Power) Solution', fontsize=16, fontweight='bold')
plt.xticks(rotation=25, fontsize=12)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bar, cost in zip(bars, costs_opf):
    plt.text(bar.get_x() + bar.get_width() / 2, 
             bar.get_height() + 0.5, 
             f'{cost:.2f} M€', 
             ha='center', fontsize=12, fontweight='bold', color='black')
plt.show()

# 3. SOLUCIÓN NSGA II - SIN COMPENSACIÓN (variable power)
# Extraer los valores del arreglo para el caso SIN COMPENSACIÓN
vol_n, n_cables_n, react1_bi_n, react2_bi_n, react3_bi_n, react4_bi_n, react5_bi_n, \
react1_val_n, react2_val_n, react3_val_n, react4_val_n, react5_val_n, S_rtr_n, p_var_n = x_nosh_var

# Calcular el costo sin compensación usando p_var_n
cost_invest_no, cost_tech_no, cost_full_no = costac_2(vol_n, n_cables_n, 
    react1_bi_n, react2_bi_n, react3_bi_n, react4_bi_n, react5_bi_n, 
    react1_val_n, react2_val_n, react3_val_n, react4_val_n, react5_val_n, 
    S_rtr_n, p_var_n)

# Extraer los costos específicos
c_vol, c_curr, c_losses, c_react, cost_tech, c_cab, c_gis, c_tr, \
c_reac, cost_invest, c_volover, c_volunder, c_ss, average_v = cost_full_no
costs_no = [c_losses, c_cab, c_gis, c_tr, c_reac, c_ss]

# GRÁFICA 2 - SIN COMPENSACIÓN
sns.set_style("darkgrid")
plt.figure(figsize=(10, 6))
colors = sns.color_palette("magma", len(costs_no))
bars = plt.bar(labels, costs_no, color=colors, edgecolor='black', linewidth=1.2)
plt.ylabel('Cost [M€]', fontsize=14, fontweight='bold')
plt.xlabel('Categories', fontsize=14, fontweight='bold')
plt.title('Breakdown of Costs Without Compensation (Variable Power)', fontsize=16, fontweight='bold')
plt.xticks(rotation=25, fontsize=12)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bar, cost in zip(bars, costs_no):
    plt.text(bar.get_x() + bar.get_width() / 2, 
             bar.get_height() + 0.5, 
             f'{cost:.2f} M€', 
             ha='center', fontsize=12, fontweight='bold', color='black')
plt.show()

# 4. GRÁFICA 3 - COMPARACIÓN APILADA ENTRE COMPENSACIÓN y SIN COMPENSACIÓN
sns.set_style("whitegrid")
ordered_labels = ['Cables', 'Substation', 'Power losses', 'Transformers', 'Switchgears', 'Reactors']
# Reordenar los datos según el nuevo orden de etiquetas
costs_opf_reordered = [costs_opf[labels.index(lbl)] for lbl in ordered_labels]
costs_no_reordered = [costs_no[labels.index(lbl)] for lbl in ordered_labels]
cumulative_opf = np.cumsum([0] + costs_opf_reordered[:-1])
cumulative_no = np.cumsum([0] + costs_no_reordered[:-1])
fig, ax = plt.subplots(figsize=(10, 6))
colors = sns.color_palette("Set2", len(ordered_labels))
for i, cost in enumerate(costs_opf_reordered):
    ax.bar('NSGA-II (Variable)', cost, bottom=cumulative_opf[i], color=colors[i], 
           edgecolor='black', linewidth=1.5, label=ordered_labels[i])
for i, cost in enumerate(costs_no_reordered):
    ax.bar('No Compensation (Variable)', cost, bottom=cumulative_no[i], color=colors[i], 
           edgecolor='black', linewidth=1.5)
# Agregar etiquetas de valores para algunas categorías (por ejemplo, Power losses)
selected_labels = ['Power losses']
for i, (cost, bottom) in enumerate(zip(costs_opf_reordered, cumulative_opf)):
    if ordered_labels[i] in selected_labels:
        text = ax.text(x=0, y=bottom + cost / 2, s=f'{cost:.2f} M€', ha='center', 
                       fontsize=12, fontweight='bold', color='black')
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])
for i, (cost, bottom) in enumerate(zip(costs_no_reordered, cumulative_no)):
    if ordered_labels[i] in selected_labels:
        text = ax.text(x=1, y=bottom + cost / 2, s=f'{cost:.2f} M€', ha='center', 
                       fontsize=12, fontweight='bold', color='black')
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])
ax.set_ylabel('Cost [M€]', fontsize=16, fontweight='bold')
ax.set_title('Total Cost Comparison (Variable Power)', fontsize=18, fontweight='bold')
plt.xticks(rotation=0, fontsize=14)
max_value = max(sum(costs_opf_reordered), sum(costs_no_reordered))
ax.set_ylim(0, max_value * 1.1)
handles, labs = ax.get_legend_handles_labels()
unique_labels = dict(zip(labs, handles))
ax.legend(unique_labels.values(), unique_labels.keys(), title="Cost Components", 
          bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, frameon=True, fancybox=True)
plt.tight_layout()
plt.show()













# plots_comparison.py - Comparación de resultados: Fixed Power vs Variable Power
# Con solo dos colores: uno para la potencia fija y otro para la potencia variable

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as path_effects

from windopti import MixedVariableProblem   # Optimización con potencia fija
from var_windopti import MixedVariableProblem  # Optimización con potencia variable
from costac_2 import costac_2

# =============================================================================
# 1. OPTIMIZACIÓN CON POTENCIA FIJA (según plots.py)
# =============================================================================

# Definir la potencia fija (p_owf = 5, es decir, 500 MW)
p_owf = 5

# --- Caso CON COMPENSACIÓN (Fija) ---
x_opf = np.array([
    1,   # react1_bi
    1,   # react2_bi
    0,   # react3_bi
    1,   # react4_bi
    0,   # react5_bi
    3,   # vol_level
    2,   # n_cables
    581.72e6,  # S_rtr
    0.944,  # react1_val
    0.845,  # react2_val
    0.0,    # react3_val
    0.892,  # react4_val
    0.0     # react5_val
])
react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, vol, n_cables, S_rtr, \
react1_val, react2_val, react3_val, react4_val, react5_val = x_opf

# Calcular costos para solución con compensación (potencia fija)
_, _, cost_fullopf = costac_2(vol, n_cables, 
    react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, 
    react1_val, react2_val, react3_val, react4_val, react5_val, 
    S_rtr, p_owf)

# Extraer componentes de costo (ejemplo: Power losses, Cables, Switchgears, Transformers, Reactors, Substation)
_, _, c_losses_fixed, _, _, c_cab_fixed, c_gis_fixed, c_tr_fixed, _, _, _, _, c_ss_fixed, _ = cost_fullopf
costs_fixed = [c_losses_fixed, c_cab_fixed, c_gis_fixed, c_tr_fixed, 0, c_ss_fixed]

# --- Caso SIN COMPENSACIÓN (Fija) ---
x_nosh = np.array([
    3,   # vol_level
    2,   # n_cables
    0,   # react1_bi
    0,   # react2_bi
    0,   # react3_bi
    0,   # react4_bi
    0,   # react5_bi
    0.0, # react1_val
    0.0, # react2_val
    0.0, # react3_val
    0.0, # react4_val
    0.0, # react5_val
    509.72e6   # S_rtr
])
vol_n, n_cables_n, react1_bi_n, react2_bi_n, react3_bi_n, react4_bi_n, react5_bi_n, \
react1_val_n, react2_val_n, react3_val_n, react4_val_n, react5_val_n, S_rtr_n = x_nosh

# Calcular costos sin compensación (potencia fija)
_, _, cost_full_no = costac_2(vol_n, n_cables_n, 
    react1_bi_n, react2_bi_n, react3_bi_n, react4_bi_n, react5_bi_n, 
    react1_val_n, react2_val_n, react3_val_n, react4_val_n, react5_val_n, 
    S_rtr_n, p_owf)

_, _, c_losses_fixed_no, _, _, c_cab_fixed_no, c_gis_fixed_no, c_tr_fixed_no, _, _, _, _, c_ss_fixed_no, _ = cost_full_no
costs_fixed_no = [c_losses_fixed_no, c_cab_fixed_no, c_gis_fixed_no, c_tr_fixed_no, 0, c_ss_fixed_no]

# =============================================================================
# 2. OPTIMIZACIÓN CON POTENCIA VARIABLE (según plots_var.py)
# =============================================================================

# --- Caso CON COMPENSACIÓN (Variable) ---
# Se incluye al final el valor de potencia variable (p_var)
x_opf_var = np.array([
    1,   # react1_bi
    1,   # react2_bi
    0,   # react3_bi
    1,   # react4_bi
    0,   # react5_bi
    3,   # vol_level
    2,   # n_cables
    581.72e6,  # S_rtr
    0.944,  # react1_val
    0.845,  # react2_val
    0.0,    # react3_val
    0.892,  # react4_val
    0.0,    # react5_val
    4.3    # p_var (valor de ejemplo)
])
react1_bi_v, react2_bi_v, react3_bi_v, react4_bi_v, react5_bi_v, vol_v, n_cables_v, S_rtr_v, \
react1_val_v, react2_val_v, react3_val_v, react4_val_v, react5_val_v, p_var = x_opf_var

# Calcular costos para solución con compensación (variable)
_, _, cost_fullopf_var = costac_2(vol_v, n_cables_v, 
    react1_bi_v, react2_bi_v, react3_bi_v, react4_bi_v, react5_bi_v, 
    react1_val_v, react2_val_v, react3_val_v, react4_val_v, react5_val_v, 
    S_rtr_v, p_var)

_, _, c_losses_var, _, _, c_cab_var, c_gis_var, c_tr_var, _, _, _, _, c_ss_var, _ = cost_fullopf_var
costs_var = [c_losses_var, c_cab_var, c_gis_var, c_tr_var, 0, c_ss_var]

# --- Caso SIN COMPENSACIÓN (Variable) ---
x_nosh_var = np.array([
    3,   # vol_level
    2,   # n_cables
    0,   # react1_bi
    0,   # react2_bi
    0,   # react3_bi
    0,   # react4_bi
    0,   # react5_bi
    0.0, # react1_val
    0.0, # react2_val
    0.0, # react3_val
    0.0, # react4_val
    0.0, # react5_val
    509.72e6,  # S_rtr
    4.1    # p_var (ejemplo)
])
vol_n_v, n_cables_n_v, react1_bi_n_v, react2_bi_n_v, react3_bi_n_v, react4_bi_n_v, react5_bi_n_v, \
react1_val_n_v, react2_val_n_v, react3_val_n_v, react4_val_n_v, react5_val_n_v, S_rtr_n_v, p_var_n = x_nosh_var

# Calcular costos sin compensación (variable)
_, _, cost_full_no_var = costac_2(vol_n_v, n_cables_n_v, 
    react1_bi_n_v, react2_bi_n_v, react3_bi_n_v, react4_bi_n_v, react5_bi_n_v, 
    react1_val_n_v, react2_val_n_v, react3_val_n_v, react4_val_n_v, react5_val_n_v, 
    S_rtr_n_v, p_var_n)

_, _, c_losses_var_no, _, _, c_cab_var_no, c_gis_var_no, c_tr_var_no, _, _, _, _, c_ss_var_no, _ = cost_full_no_var
costs_var_no = [c_losses_var_no, c_cab_var_no, c_gis_var_no, c_tr_var_no, 0, c_ss_var_no]

# =============================================================================
# 3. COMPARACIÓN DE DATOS: GRÁFICOS COMPARATIVOS
# =============================================================================

labels = ['Power losses', 'Cables', 'Switchgears', 'Transformers', 'Reactors', 'Substation']

# Función auxiliar para etiquetar las barras
def autolabel(bars, ax):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),  # Desplazamiento vertical
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

# Definir los dos colores: uno para Fixed y otro para Variable
fixed_color = "blue"
variable_color = "red"
width = 0.35
x = np.arange(len(labels))

# ------------------------------
# Gráfico 1: Comparación CON COMPENSACIÓN
# ------------------------------
sns.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(12, 6))

bars_fixed = ax.bar(x - width/2, costs_fixed, width, label='Fixed Power', color=fixed_color)
bars_var   = ax.bar(x + width/2, costs_var, width, label='Variable Power', color=variable_color)

ax.set_ylabel('Cost [M€]', fontsize=14, fontweight='bold')
ax.set_title('Comparison of Cost Breakdown WITH Compensation', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, fontsize=12)
ax.legend(fontsize=12)

autolabel(bars_fixed, ax)
autolabel(bars_var, ax)
plt.tight_layout()
plt.show()

# ------------------------------
# Gráfico 2: Comparación SIN COMPENSACIÓN
# ------------------------------
sns.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(12, 6))

bars_fixed_no = ax.bar(x - width/2, costs_fixed_no, width, label='Fixed Power', color=fixed_color)
bars_var_no   = ax.bar(x + width/2, costs_var_no, width, label='Variable Power', color=variable_color)

ax.set_ylabel('Cost [M€]', fontsize=14, fontweight='bold')
ax.set_title('Comparison of Cost Breakdown WITHOUT Compensation', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, fontsize=12)
ax.legend(fontsize=12)

autolabel(bars_fixed_no, ax)
autolabel(bars_var_no, ax)
plt.tight_layout()
plt.show()

# ------------------------------
# Resumen Comparativo (Total de Costos)
# ------------------------------
total_fixed_with = sum(costs_fixed)
total_var_with   = sum(costs_var)
total_fixed_without = sum(costs_fixed_no)
total_var_without   = sum(costs_var_no)

print("Total Cost Comparison:")
print(" WITH Compensation:")
print(f"  Fixed Power:    {total_fixed_with:.2f} M€")
print(f"  Variable Power: {total_var_with:.2f} M€")
print(" WITHOUT Compensation:")
print(f"  Fixed Power:    {total_fixed_without:.2f} M€")
print(f"  Variable Power: {total_var_without:.2f} M€")
