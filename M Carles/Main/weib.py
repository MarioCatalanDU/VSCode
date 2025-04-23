import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import gamma
import matplotlib.colors as mcolors

##############################################################################
# 1. CONFIGURACIÓN INICIAL
##############################################################################
season = "normal"       # Estación (ej. "invierno" o "verano")
N = 10000                 # Número de muestras (simulación de velocidades)
k = 1.5                   # Parámetro de forma de la distribución Weibull
lambda_ = 10              # Parámetro base de escala (m/s)

# Ajuste de lambda según la estación
if season == "invierno":
    lambda_ *= 1.2  # viento más fuerte
elif season == "normal":
    lambda_ *= 1
elif season == "verano":
    lambda_ *= 0.8  # viento más débil

seed = 42                 # Semilla para reproducibilidad

n_turbines = 36           # Número de turbinas del parque
rated_power = 500         # Potencia nominal total del parque (MW)

# Parámetros para la curva de potencia
v_cut_in = 3              # Velocidad mínima para generar potencia (m/s)
v_nominal = 12            # Velocidad nominal: a partir de aquí se alcanza la potencia nominal
v_corte = 25              # Velocidad de corte: para v ≥ 25, la turbina se apaga (P = 0)
P_rated_turbine = 500e6 / n_turbines   # Potencia nominal por turbina (~13.89e6 W)
rho = 1.225               # Densidad del aire (kg/m³)
Cp = 0.45                 # Coeficiente de potencia (valor realista)
A = 38000                 # Área del rotor (m²)

##############################################################################
# 2. FUNCIONES
##############################################################################
def generate_wind_speeds_weibull(N, k, lam, seed=None):
    if seed is not None:
        np.random.seed(seed)
    wind_speeds = np.random.weibull(k, N) * lam
    probs = (k/lam) * (wind_speeds/lam)**(k-1) * np.exp(-(wind_speeds/lam)**k)
    probs /= np.sum(probs)
    return wind_speeds, probs

def wind_speed_to_power(wind_speeds, rho=1.225, Cp=0.45, A=38000,
                        n_turbines=36, v_cut_in=3, v_nominal=12, v_corte=25, 
                        P_rated_turbine=13.89e6):
    power_per_turbine = np.zeros_like(wind_speeds)
    mask_cubic = (wind_speeds >= v_cut_in) & (wind_speeds < v_nominal)
    power_cubic = 0.5 * rho * A * Cp * (wind_speeds[mask_cubic]**3)
    power_cubic = np.minimum(power_cubic, P_rated_turbine)
    power_per_turbine[mask_cubic] = power_cubic
    mask_nominal = (wind_speeds >= v_nominal) & (wind_speeds < v_corte)
    power_per_turbine[mask_nominal] = P_rated_turbine
    total_power = power_per_turbine * n_turbines
    return total_power

def calculate_capacity_factor(mean_power_mw, rated_power_mw):
    return (mean_power_mw / rated_power_mw) * 100

##############################################################################
# 3. GENERAR DATOS Y REALIZAR CÁLCULOS
##############################################################################
wind_speeds, wind_probs = generate_wind_speeds_weibull(N, k, lambda_, seed)

mean_wind_speed = np.mean(wind_speeds)
median_wind_speed = np.median(wind_speeds)
theoretical_mean = lambda_ * gamma(1 + 1/k)
sum_wind_probs = np.sum(wind_probs)

P_parque = 500e6
rho_carles = 1.225
Cp_carles = 0.4
A_carles = 38700
P_turbina = P_parque / n_turbines
v_carles = ((2 * P_turbina) / (rho_carles * A_carles * Cp_carles))**(1/3)

power_generated_w = wind_speed_to_power(
    wind_speeds,
    rho=rho,
    Cp=Cp,
    A=A,
    n_turbines=n_turbines,
    v_cut_in=v_cut_in,
    v_nominal=v_nominal,
    v_corte=v_corte,
    P_rated_turbine=P_rated_turbine
)
power_generated_mw = power_generated_w / 1e6

mean_power_generated = np.mean(power_generated_mw)
median_power_generated = np.median(power_generated_mw)
capacity_factor = calculate_capacity_factor(mean_power_generated, rated_power)

##############################################################################
# 4. MOSTRAR RESULTADOS EN PANTALLA
##############################################################################
print("========== RESULTADOS ==========")
print(f"Velocidad media simulada: {mean_wind_speed:.2f} m/s")
print(f"Velocidad mediana simulada: {median_wind_speed:.2f} m/s")
print(f"Velocidad media teórica (Weibull): {theoretical_mean:.2f} m/s")
print(f"Velocidad de Carles (ref.): {v_carles:.2f} m/s")
print(f"Suma de probabilidades Weibull (debe ser 1): {sum_wind_probs:.5f}")
print("--------------------------------")
print(f"Potencia media generada: {mean_power_generated:.2f} MW")
print(f"Potencia mediana generada: {median_power_generated:.2f} MW")
print(f"Factor de Potencia (FP): {capacity_factor:.2f}%")
print("================================")



##############################################################################
# 5. GRÁFICA 1: DISTRIBUCIÓN DE VELOCIDAD (WEIBULL)
##############################################################################
plt.figure(figsize=(10,6))
sns.histplot(wind_speeds, bins=30, kde=True, color='royalblue', alpha=0.6, label='Distribución generada')
plt.axvline(mean_wind_speed, color='red', linestyle='-.', label=f'Media: {mean_wind_speed:.2f} m/s')
plt.axvline(median_wind_speed, color='green', linestyle='-.', label=f'Mediana: {median_wind_speed:.2f} m/s')
plt.axvline(v_carles, color='purple', linestyle=':', label=f'Velocidad Carles: {v_carles:.2f} m/s')
plt.xlabel('Velocidad del viento [m/s]', fontsize=14)
plt.ylabel('Densidad de probabilidad', fontsize=14)
plt.title(f'Distribución de Weibull - {season.capitalize()} en la Costa Mediterránea', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

##############################################################################
# 6. GRÁFICA 2: DISTRIBUCIÓN DE POTENCIA (PRODUCCIÓN ACTIVA) EN RANGO 0 A 600 MW
##############################################################################
# Para visualizar la producción activa, se filtran los datos:
# Se excluyen los valores 0 (de v < v_cut_in o v ≥ v_corte)
# y se excluyen los valores que sean igual a la potencia nominal (500 MW), ya que no
# tendría sentido mostrarlos en la distribución.
mask = (power_generated_mw > 0) & (power_generated_mw < P_rated_turbine)
filtered_power = power_generated_mw[mask]

plt.figure(figsize=(10,6))
sns.histplot(filtered_power, bins=30, kde=True, color='darkorange', alpha=0.6, label='Distribución generada')
plt.plot([], [], ' ', label=f'FP: {capacity_factor:.2f}%')
plt.axvline(np.mean(filtered_power), color='red', linestyle='-.', label=f'Media: {np.mean(filtered_power):.2f} MW')
plt.axvline(np.median(filtered_power), color='green', linestyle='-.', label=f'Mediana: {np.median(filtered_power):.2f} MW')
# La línea de "Potencia Carles" en 500 MW se añade como referencia (aunque estos datos se filtran)
plt.axvline(500, color='purple', linestyle=':', linewidth=2.5, label='Potencia Carles: 500.00 MW')
plt.xlabel('Potencia generada [MW]', fontsize=14)
plt.ylabel('Densidad de probabilidad', fontsize=14)
plt.title(f'Distribución de Potencia Generada - {season.capitalize()} en la Costa Mediterránea', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, frameon=True)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0,510)  # Eje X de 0 a 600 MW
plt.show()


##############################################################################
# 7. GRÁFICA 3: POTENCIA GENERADA vs VELOCIDAD DEL VIENTO
##############################################################################

# Rango de velocidades (0, 1, 2, …, 27)
vel_range = np.arange(0, 28, 1)

# Calcular la potencia en MW
power_curve_w = wind_speed_to_power(
    vel_range,
    rho=rho,
    Cp=Cp,
    A=A,
    n_turbines=n_turbines,
    v_cut_in=v_cut_in,
    v_nominal=v_nominal,
    v_corte=v_corte,
    P_rated_turbine=P_rated_turbine
)
power_curve_mw = power_curve_w / 1e6

# Colores: Relleno semitransparente y bordes negros opacos
facecolor = mcolors.to_rgba('darkorange', alpha=0.6)
edgecolor = mcolors.to_rgba('black', alpha=1)

plt.figure(figsize=(10,6))
ax = plt.gca()
ax.set_axisbelow(True)
plt.grid(True, linestyle='--', alpha=0.7, zorder=0)

# Dibujar el histograma (barras)
plt.bar(
    vel_range,               # Se dibujan en sus posiciones originales
    power_curve_mw,
    width=1,
    align='edge',
    color=facecolor,
    edgecolor=edgecolor,
    linewidth=1,
    zorder=3
)

# Línea de evolución, centrada en cada barra (+0.5)
plt.plot(
    vel_range + 0.5,
    power_curve_mw,
    color='darkorange',
    linewidth=2,
    zorder=4
)

# Líneas de referencia
plt.axvline(mean_wind_speed, color='red', linestyle=':', linewidth=2, zorder=5,
            label=f'V.media: {mean_wind_speed:.2f} m/s')
plt.axvline(v_carles, color='purple', linestyle=':', linewidth=2, zorder=5,
            label=f'V.Carles: {v_carles:.2f} m/s')
plt.axhline(mean_power_generated, color='red', linestyle='--', linewidth=1.5, zorder=5,
            label=f'P.media: {mean_power_generated:.2f} MW')
plt.axhline(500, color='purple', linestyle='--', linewidth=1.5, zorder=5,
            label='P.Carles: 500.00 MW')

# Configuración de ejes y título
plt.xlabel('Velocidad del viento [m/s]', fontsize=14)
plt.ylabel('Potencia generada [MW]', fontsize=14)
plt.title('Potencia Generada vs. Velocidad del Viento', fontsize=16, fontweight='bold')

# Dejar un margen a la izquierda: el primer valor se dibuja a partir de 0, pero el eje X inicia en -0.5
plt.xlim(-2, 28)
plt.ylim(0, 550)
plt.xticks(np.arange(0, 28, 5))

# Leyenda situada dentro del gráfico:
# Se ubica en coordenadas de datos; en este caso, se le ancla a (0.3, 360).
plt.legend(
    fontsize=12,
    frameon=True,
    loc='lower left',
    bbox_to_anchor=(-2, 330),
    bbox_transform=ax.transData
)

plt.show()
