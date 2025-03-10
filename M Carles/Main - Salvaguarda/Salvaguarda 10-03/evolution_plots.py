# GRÁFICAS DE EVOLUCIÓN 

  # Este script simula y grafica la evolución de los costos y el comportamiento del voltaje en una red eléctrica offshore en función de diferentes niveles de generación de potencia
  # GeneraR gráficos de voltaje promedio y pérdidas de potencia para analizar el impacto de la compensación reactiva
  # COMPARACIÓN : Compensación vs No Compensación

# Se evalua
  # Costos de inversión y operación.
  # Tensiones nodales.
  # Pérdidas de potencia





# 1. IMPORTAR HERRAMIENTAS

import numpy as np                                                            # Importa la biblioteca NumPy para cálculos numéricos
import matplotlib as mpl                                                      # Biblioteca para configuraciones avanzadas de gráficos
import matplotlib.pyplot as plt                                               # Biblioteca para graficar

# Importación de Nuestras funciones de optimización
import costac_2                                                               # Importa la función de costos del sistema





# 2. INICIALIZAR DATOS

trials = 100                                                                  # Número de simulaciones a realizar
ff = costac_2.costac_2                                                                 # Se define la función de costos a usar

# Inicialización de matrices para almacenar datos
random_check = np.zeros((trials,6))                                           # Para resultados sin compensación
random_checkpf = np.zeros((trials,6))                                         # Para resultados con compensación
avg_list = np.zeros(trials)                                                   # Para almacenar el voltaje promedio sin compensación
avg_listpf = np.zeros(trials)                                                 # Para almacenar el voltaje promedio con compensación


# Dimensiones y límites del problema
d = 13                                                                        # Número de variables de decisión
num_int = 7                                                                   # Número de variables enteras en la optimización

# Definición de límites de búsqueda
lb = np.array([3, 2, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 450e6])          # Límite inferior
ub = np.array([3, 2, 1, 1, 1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1000e6])         # Límite superior

# Lista de niveles de generación a simular (2 p.u. hasta 5 p.u.)
p_owf = 5
p_owflistpf = np.linspace(2, p_owf, trials)                                   # Para simulaciones con compensación
p_owflist = np.linspace(2, p_owf, trials)                                     # Para simulaciones sin compensación

# Inicialización de historial de variables de decisión
x_history = np.zeros((trials, d))

# Configuraciones iniciales para la simulación

# Con compensación
x_opf = np.array([
        3, 
        2, 
        1, 
        1, 
        0, 
        1, 
        0, 
        0.519, 
        0.953, 
        0.0, 
        0.737, 
        0.0, 
        509.72e6
        ])

# Sin compensación
x0 = np.array([
        3, 
        2, 
        0, 
        0, 
        0, 
        0, 
        0, 
        0.0, 
        0.0, 
        0.0, 
        0.0, 
        0.0, 
        509.72e6
        ])


# Simulación CON compensación
for i in range(trials):
        x_history[i,:] = x_opf
        vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr = x_opf
        p_owf = p_owflistpf[i]
        # Se ejecuta la función de costos
        cost_investpf, cost_techpf, cost_fullpf = ff(vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr, p_owf)
        # Se almacenan los resultados
        random_checkpf[i,:] = [cost_investpf, cost_techpf, cost_fullpf[10], cost_fullpf[2], cost_fullpf[3], cost_fullpf[11]]
        # Coste promedio
        average_vpf = cost_fullpf[13]
        # Voltaje promedio
        avg_listpf[i] = average_vpf
        # Pérdidas de potencia
        cost_losses_pf = random_checkpf[:,3]

# Simulación SIN compensación
for i in range(trials):
        x_history[i,:] = x0
        vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr = x0
        p_owf = p_owflist[i]
        # Se ejecuta la función de costos
        cost_invest, cost_tech, cost_full = ff(vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr, p_owf)
        # Se almacenan los resultados
        random_check[i,:] = [cost_invest, cost_tech, cost_full[10], cost_full[2], cost_full[3], cost_full[11]]
        # Coste promedio
        average_v = cost_full[13]
        # Voltaje promedio
        avg_list[i] = average_v
        # Pérdidas de potencia
        cost_losses_no = random_check[:,3]





 # 3. GRÁFICAS
  
# Gráficos de voltaje promedio  
      
plt.plot(p_owflist, avg_list, label='Average Node Voltage no compensation')
plt.plot(p_owflist, avg_listpf, label='Average Node Voltage optimal compensation')
plt.xlabel('Power Injection [p.u]')
plt.ylabel('Average Node Voltage [p.u]')
plt.axhline(y=1.1, color='r', linestyle='--', label='Technical Constraint at 1.1 p.u.')
plt.title('Average Node Voltage at different wind conditions for an 500 MW OWPP')
plt.legend()
plt.show()



# Gráficos de pérdidas de potencia

plt.plot(p_owflist, cost_losses_no, label='Power losses with no compensation')
plt.plot(p_owflist, cost_losses_pf, label='Power losses with optimal compensation')
plt.xlabel('Power Injection [p.u]')
plt.ylabel('Power losses [M€/year]')
plt.title('Power losses at different wind conditions for a 500 MW OWPP')
plt.legend()
plt.show()


