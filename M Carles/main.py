# Este archivo main.py actúa como el punto de entrada principal para ejecutar la optimización multiobjetivo utilizando el algoritmo NSGA-III


import matplotlib.pyplot as plt                                    # matplotlib.pyplot: una de las bibliotecas más populares en Python para crear gráficos y visualizaciones. Cuando se importa como plt, se convierte en una convención ampliamente usada para trabajar con gráficos (pyplot proporciona una interfaz similar a la de MATLAB para crear gráficos de forma sencilla)
                                                                   # pymoo: Biblioteca utilizada para optimización multiobjetivo.
from pymoo.algorithms.moo.nsga3 import NSGA3                       # NSGA3: Implementación del algoritmo NSGA-III.
from pymoo.util.ref_dirs import get_reference_directions           # get_reference_directions: Genera las direcciones de referencia utilizadas por NSGA-III para guiar la búsqueda ( Esto asegura que las soluciones generadas por el algoritmo estén bien distribuidas en el frente de Pareto. Estas direcciones actúan como "puntos guía" para las soluciones optimizadas)
from pymoo.termination import get_termination                      # get_termination: definir los criterios de finalización de un algoritmo de optimización. Es una herramienta clave para especificar las condiciones bajo las cuales el proceso iterativo del algoritmo debe detenerse
from pymoo.operators.sampling.rnd import IntegerRandomSampling     # IntegerRandomSampling: método de muestreo que genera valores enteros aleatorios. Genera la población inicial asignando valores aleatorios enteros a las variables de decisión dentro de los límites permitidos. Este método asegura que todas las soluciones iniciales sean válidas y cumplan las restricciones definidas por el problema
from problem import MyProblem                                      # MyProblem: Importa MyProblem desde el archivo problem.py, que define el problema de optimización
from pymoo.optimize import minimize                                # minimize: Esta función es el motor principal para realizar la optimización en pymoo (La función minimize ejecuta el proceso de optimización. Recibe como entrada el problema que se desea resolver, el algoritmo a utilizar, las condiciones de parada y otros parámetros. Su tarea es resolver el problema de optimización y devolver los resultados.)
from pymoo.operators.crossover.sbx import SBX                      # SBX: Crossover (para crear población Qt)
from pymoo.operators.mutation.pm import PM                         # PM: Mutation (para crear población Qt)
from pymoo.operators.repair.rounding import RoundingRepair         # RoundingRepair: (arreglos en variables)
from pymoo.visualization.scatter import Scatter                    # Scatter: Visualización gráfica
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival     # RankAndCrowdingSurvival: utilizada en el proceso de selección de supervivencia durante la optimización. Implementa (1. Non-Dominated Sorting (Clasificación No-Dominada) "Ordena las soluciones en diferentes frentes de Pareto basándose en su dominancia" // 2. Crowding Distance (Distancia de Apiñamiento) "Dentro de cada frente, prioriza soluciones que están más "alejadas" unas de otras en el espacio de los objetivos, Favorece la diversidad en la población")
from pymoo.core.mixed import MixedVariableGA                       # MixedVariableGA: implementación de un algoritmo genético (GA) que maneja variables mixtas (enteras, continuas, binarias)



# STEP 1: 
 # Generate reference directions for NSGA-III
 # These reference directions help the algorithm distribute solutions evenly across the objective space
  # 1. Las soluciones estén bien distribuidas
  # 2. Cada solución esté lo más cerca posible de una dirección de referencia, ayudando al algoritmo a explorar todo el frente de Pareto
# Sintaxis: get_reference_directions(method, n_dim, n_partitions, **kwargs)
   # method: Especifica el método para generar las direcciones de referencia
   # n_dim: Es el número de dimensiones u objetivos del problema multiobjetivo. En el caso del código, es 2 porque se trata de un problema biobjetivo
   # n_partitions: Especifica cómo dividir el espacio objetivo. Un número mayor genera más direcciones y una distribución más densa
   # **kwargs: -
ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=50)



# STEP 2: 
 # Inicializa el algoritmo NSGA-III
# Sintaxis: 
  # pop_size: Tamaño de la población: Define cuántas soluciones (individuos) hay en cada generación. Un tamaño grande permite explorar más ampliamente el espacio de soluciones, pero incrementa el costo computacional
  # ref_dirs = ref_dirs: Define las direcciones de referencia generadas previamente con get_reference_directions
  # sampling: Method to initialize the population (IntegerRandomSampling genera valores aleatorios para las variables de decisión, asegurando que sean enteros)
  # crossover and mutation: Operators to generate new solutions (Crossover: combina soluciones existentes para generar nuevas // Mutation: introduce pequeñas alteraciones en las soluciones para explorar nuevas regiones del espacio de búsqueda)
  # repair: Ensures solutions respect variable constraints
algorithm = NSGA3(
    pop_size=800,                                                              # A large population size to explore the solution space
    ref_dirs=ref_dirs,                                                         # Reference directions defined earlier
    sampling=IntegerRandomSampling(),                                          # Randomly initialize population (integer sampling)
    crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),    # SBX:(Simulated Binary Crossover) // (prob=1.0: Aplica el cruce al 100% de los individuos // eta=3.0: Controla qué tan lejos están los hijos de los padres en términos de valores de las variables. Valores más altos generan hijos más cercanos a los padres // vtype=float: Especifica que las variables resultantes deben ser de tipo flotante //  repair=RoundingRepair(): Asegura que las soluciones generadas sean válidas al redondear valores fuera de los límites permitidos)
    mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),      # PM:(Polynomial Mutation) // (prob=1.0: Aplica la mutación a todas las soluciones // eta=3.0: Controla la magnitud del cambio introducido por la mutación // vtype=float y repair=RoundingRepair(): Igual que en el cruce, asegura que las soluciones mutadas sean válidas)
)



# STEP 3: 
 # Define termination criteria
# Sintaxis: get_termination(criterion, value, **kwargs)
  # criterion: porque parará el código (n_gen, time, f_tol...) 
  # value: valor específico asociado al criterio
termination = get_termination("n_gen", 50)    # The algorithm stops after 50 generations (iterations)



# STEP 4: 
 # Optimization process
# Sintaxis:
  # problem: Define el problema de optimización que se desea resolver 
  # algorithm: El algoritmo que se usará para resolver el problema (por ejemplo, NSGA-II, NSGA-III)
  # termination: Define los criterios de parada del algoritmo
  # seed: Establece una semilla para asegurar reproducibilidad. Si se fija un valor, los resultados serán consistentes en cada ejecución
  # save_history: Si es True, almacena el historial de las generaciones y permite análisis posteriores de las soluciones intermedias
  # verbose: Si es True, imprime información sobre el progreso durante la optimización
res = minimize(
    MyProblem(),          # Problema definido en "problem.py"
    algorithm,            # Algoritmo NSGA-III configurado
    termination,          # Criterio de parada 
    seed=1,               # Semilla para reproducibilidad
    save_history=True,    # Guarda el historial
    verbose=True          # Muestra el progreso
)
# Devuelve:
  # X: Las soluciones óptimas encontradas (valores de las variables de decisión).
  # F: Los valores correspondientes de las funciones objetivo para cada solución en X.
  # history: Si save_history=True, contiene un registro de todas las generaciones y poblaciones intermedias.
  # algorithm: El estado final del algoritmo después de la optimización.
  # n_eval: El número total de evaluaciones de la función objetivo realizadas.



# STEP 5: 
 # Display results
 # Print the best solutions found (decision variables and objective values)
print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))


# STEP 6:
 # Print graph
 # Visualize the Pareto front
 # The scatter plot shows the trade-off between the two objective functions
plt.scatter(res.F[:, 0], res.F[:, 1])    # Crea un gráfico de dispersión con los valores de las funciones objetivo (objetivo 1 frente a objetivo 2) de las soluciones encontradas
plt.title("Pareto Front")                # Agrega un título al gráfico
plt.xlabel("Objective 1")                # Etiqueta el eje X 
plt.ylabel("Objective 2")                # Etiqueta el eje Y
plt.show()                               # Renderiza el gráfico en la pantall
