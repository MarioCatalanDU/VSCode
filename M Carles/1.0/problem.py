# Este archivo es el núcleo del problema de optimización. Define cómo se calculan los objetivos, las restricciones y cómo se simula la red eléctrica offshore.
 # Define:
   # Las variables y restricciones del problema     (return Y_bus, p_owf, q_owf, n_cables, u_i, I_rated, S_rtr, Y_l1, Y_l2, Y_l3, Y_l4, Y_l5, A, B, C, Y_trserie, Y_piserie)
   # El flujo de potencia mediante Newton-Raphson   (return V_wslack, angle_wslack, curr, p_wslack, q_wslack, solution_found)
   # La evaluación de costos y restricciones        (return cost_invest, cost_tech, gs)



import numpy as np                                    # numpy: Librería. Se utiliza para operaciones matemáticas y de álgebra lineal
from pymoo.core.problem import ElementwiseProblem     # ElementwiseProblem: indica que cada solución se evalúa individualmente
import cmath                                          # Librería. Biblioteca matemática para números complejos





# DEFINE the optimization problem with mixed variable types
class MyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=13,                                                                             # n_var: Número de variables de decisión 
            n_obj=2,                                                                              # n_obj: Número de funciones objetivo (2: costos de inversión y técnicos)
            n_constr=14,  # change if needed                                                      # n_constr: Número de restricciones, que aseguran que las soluciones sean válidas
            xl=np.array([0, 0, 0, 0, 0, 1, 2, 200e6, 0.0, 0.0, 0.0, 0.0, 0.0]),                   # xl: Límites inferiores  para cada variable de decisión
            xu=np.array([1, 1, 1, 1, 1, 2, 3, 800e6, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]),    # xu: Límites superiores para cada variable de decisión
            type_var=np.array([int, int, int, int, int, int, int,                                 # type_var: Especifica los tipos de las variables (enteras o continuas)
                               float, float, float, float, float, float]) 
        )






    # EVALUATE
    # evalúa las soluciones generadas por el algoritmo de optimización
    def _evaluate(self, x, out, *args, **kwargs):

        # VARIABLES DE DECISIÓN: Estas representan las decisiones que el algoritmo puede ajustar
        react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, vol, n_cables, S_rtr, react1, react2, react3, react4, react5 = x      
          # react1_bi a react5_bi: Binarias, indican si hay un reactor instalado en una posición, si los reactores están activados (1) o desactivados (0)
          # vol: Nivel de voltaje de la red (1 para 132 kV, 2 para 220 kV)
          # n_cables: Número de cables en paralelo
          # S_rtr: Potencia nominal del transformador
          # react1 a react5: Tamaños de los reactores

        # Tamaños de los reactores
        react1 /= 1000                                                                                                             
        react2 /= 1000                                                                                                             
        react3 /= 1000                                                                                                             
        react4 /= 1000                                                                                                             
        react5 /= 1000





        # Evaluate.1 
         # CONSTRUCCIÓN DEL MODELO DE RED (build_grid_data)
          # Build the admittance matrix (Ybus) of the grid and define the general data
    
        # Parámetros de entrada:
           # Datos base: Sbase, f, l
           # Propiedades del sistema: vol, S_rtr (potencia del transformador), n_cables
        def build_grid_data(
                            Sbase,         # Sbase: Base power of the grid
                             f,            # f: Frequency of the grid
                             l,            # l: Length of the cables
                             p_owf,        # p_owf: Active power of the offshore wind farm
                             q_owf,        # q_owf: Reactive power of the offshore wind farm
                             vol,          # vol: Voltage level of the grid
                             S_rtr,        # S_rtr: Rated power of the transformer
                             n_cables,     # n_cables: Number of cables
                             react1_bi, react2_bi, react3_bi, react4_bi, react5_bi,        # reactX_bi: Binary value for the X compensator 
                             react1_val, react2_val, react3_val, react4_val, react5_val    # reactX_val: Value of the X compensator
                             ):
          # return Y_bus, p_owf, q_owf, n_cables, u_i, I_rated, S_rtr, Y_l1, Y_l2, Y_l3, Y_l4, Y_l5, A, B, C, Y_trserie, Y_piserie    
           

            # Propiedades de los cables: Define los valores de resistencia (R), capacitancia (Cap) e inductancia (L) dependiendo del nivel de voltaje (vol)
            if vol == 1:       # Vol = 1 -> 132 kv
                u_i = 132e3    # V
                R = 0.0067     # ohm/km
                Cap = 0.19e-6  # F/km
                L = 0.38e-3    # H/km
                A = 1.971e6
                B = 0.209e6
                C = 1.66
                I_rated = 500  # A

            if vol == 2:       # Vol = 2 -> 220 kv
                u_i = 220e3    # V
                R = 0.0067     # ohm/km
                Cap = 0.17e-6  # F/km
                L = 0.40e-3    # H/km
                A = 3.181e6
                B = 0.11e6
                C = 1.16
                I_rated = 540  # A

            
            Y_ref = Sbase / u_i**2  # 1 / ohm
            V_ref = u_i

            # 1.2 TRAFO
             # Trafo parameters
            U_rtr = u_i  # V
            P_Cu = 60e3  # W
            P_Fe = 40e3  # W
            u_k = 0.18   # p.u.
            i_o = 0.012  # p.u.

            # Computation of Y parallel (fórmulas)
            G_tri = (P_Fe / U_rtr**2)
            B_tri = - (i_o * (S_rtr / U_rtr**2))
            Y_tr = (G_tri + 1j * B_tri) / Y_ref

            # Computation of Y series (fórmulas)
            R_tr = P_Cu / S_rtr
            X_tr = np.sqrt((u_k * (U_rtr**2 / S_rtr))**2 - R_tr**2)
            Z_tr = R_tr + 1j * X_tr
            Y_trserie =  (1 / Z_tr) / Y_ref

            # 1.3 CABLES
            R = 0.0067     # ohm/km
            Cap = 0.17e-6  # F/km
            L = 0.40e-3    # H/km
            Y = 1j * (2 * np.pi * f * Cap / 2)
            Z = R + 1j * (2 * np.pi * f * L)
            theta = l / 2 * np.sqrt(Z * Y)
            Y_pi = n_cables * (Y * l / 4 * np.tanh(theta / 4) / (theta / 4)) / Y_ref
            G_pi = np.real(Y_pi)
            B_pi = np.imag(Y_pi)
            Z_piserie = (Z * l / 2 * np.sinh(theta /2) / (theta/2)) 
            Y_piserie = n_cables * (1 / Z_piserie)  / Y_ref 

            # 1.4 COMPENSATOR
            if react1_bi:
                Y_l1 = - 1j * react1_val
            else:
                Y_l1 = 0

            if react2_bi:
                Y_l2 = - 1j * react2_val
                
            else:
                Y_l2 = 0

            if react3_bi:
                Y_l3 = - 1j * react3_val
            else:
                Y_l3 = 0

            if react4_bi:
                Y_l4 = - 1j * react4_val
            else:
                Y_l4 = 0

            if react5_bi:
                Y_l5 = - 1j * react5_val
            else:
                Y_l5 = 0

            # 1.5 Grid connection
            scr = 5  # which value should we put here 5 or 50?
            xrr = 10
            zgridm = V_ref**2 / (scr * p_owf * Sbase)
            rgrid = np.sqrt(zgridm**2 / (xrr**2 + 1))
            xgrid = xrr * rgrid
            zgrid = rgrid + 1j * xgrid
            Y_g = (1 / zgrid) / Y_ref


            # Y_BUS
            Y_bus = np.array([[Y_trserie + Y_tr + Y_l1, -Y_trserie, 0, 0, 0, 0],
                        [-Y_trserie, Y_piserie + Y_pi + Y_l2 + Y_trserie, - Y_piserie, 0, 0, 0],
                        [0, -Y_piserie, 2 * Y_piserie + 2 * Y_pi + Y_l3, -Y_piserie, 0, 0],
                        [0, 0, -Y_piserie, Y_piserie + Y_pi + Y_l4 + Y_trserie, -Y_trserie, 0],
                        [0, 0, 0, -Y_trserie, Y_trserie + Y_tr + Y_l5 + Y_g, -Y_g],
                        [0, 0, 0, 0, -Y_g, Y_g]])
        
            return Y_bus, p_owf, q_owf, n_cables, u_i, I_rated, S_rtr, Y_l1, Y_l2, Y_l3, Y_l4, Y_l5, A, B, C, Y_trserie, Y_piserie
                   
          # return Y_bus, p_owf, q_owf, n_cables, u_i, I_rated, S_rtr, Y_l1, Y_l2, Y_l3, Y_l4, Y_l5, A, B, C, Y_trserie, Y_piserie    
        # FIN Evaluate.1 
        




        # Evaluate.2:
         # NEWTON-RAPHSON 
          # Usa el método de Newton-Raphson para calcular el flujo de potencia:
           # V_wslack, angle_wslack: Voltajes y ángulos en los nodos
           # p_wslack, q_wslack: Potencias activa y reactiva en el nodo slack
           # curr: Corrientes en las líneas

        def run_pf(
               p_owf: float=0.0,           # p_owf: Active power of the offshore wind farm
               q_owf: float=0.0,           # q_owf: Reactive power of the offshore wind farm
               Y_bus: np.ndarray=None,     # Y_bus: Admittance matrix of the grid
               nbus: int=1,                # nbus: Number of buses in the grid
               V_slack: float=1.0,         # V_slack: Voltage of the slack bus
               angle_slack: float=0.0,     # Angle_slack: Angle of the slack bus
               max_iter: int=20,           # max_iter: Maximum number of iterations
               eps: float=1e-6,            # eps: Error tolerance
               y_trserie: float=0.0,       # y_trserie: Admittancia serie del transformador
               y_piserie: float=0.0,       # y_piserie: Admittancia serie de la configuración en pi de los cables 
               S_rtr: float=500e6,         # S_rtr: Potencia nominal del transformador
               n_cables: int=2,            # n_cables: Número de cables en paralelo en la red
               vol: int=1                  # vol: Nivel de voltaje del sistema
               ):      
            # return: [V, V_wslack, angle_wslack, curr, p_wslack, q_wslack, solution_found]

    
            # 2.1 Inicialización de parámetros

            # Se inicializan todos los nodos (𝑉) excepto el nodo slack, cuyo valor es fijo (𝑉slack=1.0)
            V = np.ones(nbus - 1, dtype=float)
            V_wslack = np.empty(nbus, dtype=float)
            V_wslack[:nbus - 1] = V
            V_wslack[nbus - 1] = V_slack

            # Los ángulos (θ) de los nodos se inicializan en 0, excepto el nodo slack, cuyo ángulo es fijo (θ slack=0.0)
            angles = np.zeros(nbus - 1, dtype=float)
            angle_wslack = np.empty(nbus, dtype=float)
            angle_wslack[:nbus - 1] = angles
            angle_wslack[nbus - 1] = angle_slack


            # 2.2 Preparación de variables

            x0 = np.concatenate([angles, V])            # xo: Vector inicial que combina voltajes (V) y ángulos (θ)
            x = x0

            P_obj = np.array([p_owf, 0, 0, 0, 0])       # P objetivo. Se fija Powf, para el nodo del parque eólico
            Q_obj = np.array([q_owf, 0, 0, 0, 0])       # Q objetivo. Se fija Qowf, para el nodo del parque eólico
            PQ_obj = np.concatenate([P_obj, Q_obj])     # Vector combinado de P y Q, que define las potencias objetivo en cada nodo


            # 2.3 Bucle de iteración
              
              # Criterio de convergencia
                # ϵ: Error en el desajuste (Δ𝑃,Δ𝑄)
                # El bucle se detiene cuando el error es menor que el umbral ((deltaPQ) < (eps)) o se alcanza el número máximo de iteraciones

            epsilon = 1e10                              # Tolerancia
            k = 0                                       # Contador de iteraciones

            while epsilon > eps and k < max_iter:       # eps = 1e-6

                k = k + 1
                x = np.concatenate((angles, V))
                P = np.zeros(nbus - 1)
                Q = np.zeros(nbus - 1)

                # 2.3.1 Cálculo de potencias calculadas (Compute power mismatch function)

                # P
                for i in range(nbus - 1):
                    for j in range(nbus):
                        P[i] = P[i] + V_wslack[i] * V_wslack[j] * (np.real(Y_bus[i, j]) * np.cos(
                            angle_wslack[i]-angle_wslack[j]) + np.imag(Y_bus[i, j]) * np.sin(angle_wslack[i]-angle_wslack[j]))
                        
                # Q
                for i in range(nbus - 1):
                    for j in range(nbus):
                        Q[i] = Q[i] + V_wslack[i] * V_wslack[j] * (np.real(Y_bus[i, j]) * np.sin(
                            angle_wslack[i]-angle_wslack[j]) - np.imag(Y_bus[i, j]) * np.cos(angle_wslack[i]-angle_wslack[j]))


                # 2.3.2 Cálculo del desajuste (compute error in mismatch function)
                PQ = np.concatenate((P, Q))
                deltaPQ = (PQ_obj - PQ)               # deltaPQ: Vector de desajuste entre las potencias objetivo y las calculadas


                # 2.3.3 Construcción de la matriz Jacobiana (build the Jacobian)
                 # Relaciona (deltaP, deltaQ) con los cambios de Voltaje (V) y ángulo (θ)
                J11 = np.zeros((nbus - 1, nbus - 1))   # P / angle
                J12 = np.zeros((nbus - 1, nbus - 1))   # P / V
                J21 = np.zeros((nbus - 1, nbus - 1))   # Q / angle
                J22 = np.zeros((nbus - 1, nbus - 1))   # Q / V
                
                # J11
                for i in range(nbus - 1):
                    for j in range(nbus - 1):
                        if j == i:
                            J11[i, j] = - Q[i] - (V[i]**2) * np.imag(Y_bus[i, i])
                        else:
                            J11[i, j] = abs(V[i]) * abs(V[j]) * (np.real(Y_bus[i, j]) * np.sin(
                                angles[i] - angles[j]) - np.imag(Y_bus[i, j]) * np.cos(angles[i]-angles[j]))
                # J12
                for i in range(nbus - 1):
                    for j in range(nbus - 1):
                        if j == i: 
                            J12[i, j] = P[i] / abs(V[i]) + \
                                np.real(Y_bus[i, i]) * abs(V[i])
                        else:
                            J12[i, j] = abs(V[i]) * (np.real(Y_bus[i, j]) * np.cos(angles[i] -
                                angles[j]) + np.imag(Y_bus[i,j]) * np.sin(angles[i]-angles[j]))
                # J21
                for i in range(nbus - 1):
                    for j in range(nbus - 1):
                        if j == i:
                            J21[i, j] = P[i]-(V[i,])**2 * np.real(Y_bus[i, i])
                        else:
                            J21[i, j] = - abs(V[i])*abs(V[j]) * (np.real(Y_bus[i, j]) * np.cos(
                                angles[i]-angles[j]) + np.imag(Y_bus[i, j]) * np.sin(angles[i]-angles[j]))
                # J22
                for i in range(nbus - 1):
                    for j in range(nbus - 1):
                        if j == i:
                            J22[i, j] = Q[i] / abs(V[i]) - \
                                np.imag(Y_bus[i, i]) * abs(V[i])
                        else:
                            J22[i, j] = abs(V[i]) * (np.real(Y_bus[i, j]) * np.sin(angles[i] -
                                                                            angles[j]) - Y_bus[i, j].imag*np.cos(angles[i]-angles[j]))
                # J
                J = np.concatenate((np.concatenate((J11, J12), axis=1),
                                np.concatenate((J21, J22), axis=1)), axis=0)



                # 2.4 Resolución del sistema

                delta_x = np.linalg.solve(J, deltaPQ)     # delta_x: Se obtiene resolviendo el sistema lineal (JΔx=−ΔPQ)
                
                # Actualización
                # we have updated value of angles and V. [1X10] matrix (note this vector does not include slack!)
                # Las variables (𝑉 y θ) se actualizan sumando Δ𝑥
                x_new = x + delta_x  
                
                angles = x_new[0:5]
                V = x_new[5:10]
                angle_wslack[:nbus - 1] = angles
                V_wslack[:nbus - 1] = V

                # Validación de convergencia (we check error value)
                epsilon = (max(abs(deltaPQ)))    # Combprueba que (deltaPQ) < (eps)    



            # 2.5 Cálculo de resultados finales

            solution_found = False
            p_wslack = np.zeros(nbus)
            q_wslack = np.zeros(nbus)
            curr_inj = np.zeros((nbus),dtype = "complex")
            curr = np.zeros(nbus - 2)

            if k + 1 < max_iter:
                solution_found = True

                Iinj = Y_bus @ (V_wslack * np.exp(1j * angle_wslack))            # Iing:  Corrientes inyectadas en cada nodo
                Sinj = (V_wslack * np.exp(1j * angle_wslack)) * np.conj(Iinj)    # Sinj: Potencias complejas inyectadas


                #  Now we can get the P ad Q values of the slack node 6 (note that the slack node is the last one of the vectors)
                for i in range(nbus):
                        for j in range(nbus):
                            p_wslack[i] = p_wslack[i] + V_wslack[i] * V_wslack[j] * (np.real(Y_bus[i, j]) * np.cos(
                                angle_wslack[i]-angle_wslack[j]) + np.imag(Y_bus[i, j]) * np.sin(angle_wslack[i]-angle_wslack[j]))
                            
                for i in range(nbus):
                        for j in range(nbus):
                            q_wslack[i] = q_wslack[i] + V_wslack[i] * V_wslack[j] * (np.real(Y_bus[i, j]) * np.sin(
                                angle_wslack[i]-angle_wslack[j]) - np.imag(Y_bus[i, j]) * np.cos(angle_wslack[i]-angle_wslack[j]))
                            

                # Now we compute the injected currents at each node
                for i in range(nbus):
                    for j in range(nbus):
                        curr_inj[i] += Y_bus[i,j] * cmath.rect(V_wslack[j],angle_wslack[j])


                #  Se calculan corrientes específicas entre nodos  
                  #  When it comes to overcurrents, we are interested in line currents, not node injection currents.
                  #  We compute now line currents. Note they are normalized to the power of the plant
                i_21 = abs((cmath.rect(V[0],angles[0]) - cmath.rect(V[1],angles[1])) * y_trserie) / p_owf
                i_32 = abs((cmath.rect(V[1],angles[1]) - cmath.rect(V[2],angles[2])) * y_piserie) / p_owf
                i_43 = abs((cmath.rect(V[2],angles[2]) - cmath.rect(V[3],angles[3])) * y_piserie) / p_owf
                i_54 = abs((cmath.rect(V[3],angles[3]) - cmath.rect(V[4],angles[4])) * y_trserie) / p_owf

                curr = np.array([i_21, i_32, i_43, i_54])

            else:
                print("No solution found")
                pass

            return V_wslack, angle_wslack, curr, p_wslack, q_wslack, solution_found
        
          # return V_wslack, angle_wslack, curr, p_wslack, q_wslack, solution_found
        # Fin Evaluate.2





        # Evaluate.3:
         # CÁLCULOS DE COSTES
         # Calcula:
           # cost_invest: Costo de inversión 
           # cost_tech: Costo técnico
           # gs: Restricciones
           
        def compute_costs(
                    p_owf: float=0.0,              # p_owf: Active power of the offshore wind farm   
                    p_wslack: float=0.0,           # p_wslack: Active power at each node
                    q_wslack: float=0.0,           # q_wslack: Reactive power at each node
                    V: np.ndarray=None,            # V: Voltage at each node
                    curr: np.ndarray=None,         # curr: Current at each line
                    nbus: int=1,                   # nbus: Number of buses in the grid
                    n_cables: int=1,               # n_cables: Number of cables
                    u_i: float=220,                # u_i: Rated voltage of the cables
                    I_rated: float=1.0,            # I_rated: Rated current of the cables
                    S_rtr: float=500,              # S_rtr: Rated power of the transformer
                    react1_bi: bool=False,         # react1_bi: Binary value for the first compensator
                    react2_bi: bool=False,         # react2_bi: Binary value for the second compensator
                    react3_bi: bool=False,         # react3_bi: Binary value for the third compensator
                    react4_bi: bool=False,         # react4_bi: Binary value for the fourth compensator
                    react5_bi: bool=False,         # react5_bi: Binary value for the fifth compensator
                    Y_l1: float=0.0,               # Y_l1: Value of the first compensator
                    Y_l2: float=0.0,               # Y_l2: Value of the second compensator
                    Y_l3: float=0.0,               # Y_l3: Value of the third compensator
                    Y_l4: bool=False,              # Y_l4: Value of the fourth compensator
                    Y_l5: bool=False,              # Y_l5: Value of the fifth compensator
                    solution_found: bool=False     # solution_found: Boolean value indicating if a solution was found
                    ):
                #return: [cost_invest, cost_tech]

            """
            if not solution_found:
                print("NO SOLUTION !!!")
                cost_invest = 1e20
                cost_tech = 1e20
                cost_tech1 = 1e20 
                cost_tech2 = 1e20
                cost_tech3 = 1e20
                cost_tech4 = 1e20

            else:
            """
            
            # 3.1 COSTOS

            # AC power losses 
             # (coste técnico)
            p_lossac = Sbase * (p_owf + p_wslack[5]) * 1e-6  # MW

            # Cable cost 
             # (coste inversión)
            Sncab = np.sqrt(3) * u_i * I_rated
            eur_sek = 0.087  # 0.087 eur = 1 sek
            c_cab = n_cables * (A + B * np.exp(C * Sncab / 1e8)) * l * eur_sek / 1e6

            # Cost switchgears
             # (coste inversión)
            c_gis = (0.0017 * u_i * 1e-3 + 0.0231)  # u_i in kV

            # Cost susbstation
             # (coste inversión)
            c_ss = (2.534 + 0.0887 * p_owf * 100)  # p-owf in MW

            # Cost power losses
             # (coste técnico)
            t_owf = 25  # lie time in years
            c_ey = 100  # eu/MWh, cost of energy lost 
            c_losses = (8760 * t_owf * c_ey * p_lossac) / 1e6 # losses in MW , 8760 since 1 year is 8760 h

            # Cost transformers
             # (coste inversión)
            c_tr = (0.0427 * (S_rtr * 1e-6)**0.7513)  # S_rtr in MVA


            # Cost reactors (shunt reactors)
             # (coste inversión)
            fact = 1e4
            k_on = 0.01049 * fact      # Factores de costos
            k_mid = 0.01576 * fact     # Factores de costos
            k_off = 0.01576 * fact     # Factores de costos
            p_on = 0.8312              # Penalizaciones adicionales
            p_mid = 1.244              # Penalizaciones adicionales
            p_off = 1.244              # Penalizaciones adicionales

             # Cada reactor tiene costos asociados a su tamaño y ubicación
            if react1_bi:
                c_r1 = k_off * (abs(Y_l1) * (V[0])**2) + p_off
            else:
                c_r1 = 0

            if react2_bi:
                c_r2 = k_off * (abs(Y_l2) * (V[1])**2) + p_off
            else:
                c_r2 = 0

            if react3_bi:
                c_r3 = k_mid * (abs(Y_l3) * (V[2])**2) + p_mid
            else:
                c_r3 = 0

            if react4_bi:
                c_r4 = k_on * (abs(Y_l4) * (V[3])**2) + p_on
            else:
                c_r4 = 0

            if react5_bi:
                c_r5 = k_on * (abs(Y_l5) * (V[4])**2) + p_on
            else:
                c_r5 = 0
            
            # c_reac = c_r1 + c_r2 + c_r3 + c_r4 + c_r5
            c_reac = (c_r1 + c_r2 + c_r3 + c_r4 + c_r5) * 1


            # Compensación Q
             # (coste técnico)
              # we want reactive power delivered to the grid to be as close as possible to 0
            c_react = 0
            if q_wslack[nbus-1] != 0:
                    c_react = abs(q_wslack[nbus-1]) * 100
            
            # Overcurrents
             # (coste técnico)
            c_curr = 0
            for i in [1, 2]:  # check only the cable for now
                c_curr += (max(curr[i] - 1.1 * n_cables, 0)) * 100



            # 3.2 RESTRICCIONES

             # Las restricciones garantizan que el sistema opera dentro de límites aceptables
             # we try to implement the constraints in pymoo form

            # Overvoltages (Sobretensiones)
             # Verifica que el voltaje en cada nodo no supere 1.1 pu
            g1_ov = V[0] - 1.1
            g2_ov = V[1] - 1.1
            g3_ov = V[2] - 1.1
            g4_ov = V[3] - 1.1
            g5_ov = V[4] - 1.1
            # TODO: do the same for the others

            # Under voltages (Subtensiones)
             # Verifica que el voltaje no caiga por debajo de 0.9 pu
            g1_uv = 0.9 - V[0]
            g2_uv = 0.9 - V[1]
            g3_uv = 0.9 - V[2]
            g4_uv = 0.9 - V[3]
            g5_uv = 0.9 - V[4]
            # TODO: do the same for the others

            # Over current (Sobrecorreintes)
             # Limita la corriente máxima
            # Transformers  
            i_max_tr = S_rtr / Sbase                # rated current of the transformer
            g1_octr = abs(curr[0]) - i_max_tr
            g2_octr = abs(curr[3]) - i_max_tr
            # Cables
            i_maxcb =  (Sncab / Sbase) * n_cables   # rated current of the cables
            g3_oc = abs(curr[1]) - i_maxcb
            g4_oc = abs(curr[2]) - i_maxcb


            # Vector Restricciones
            gs = [g1_ov, g2_ov, g3_ov, g4_ov, g5_ov, g1_uv, g2_uv, g3_uv, g4_uv, g5_uv, g1_octr,g2_octr, g3_oc, g4_oc]
            

            cost_invest = c_cab + c_gis + c_tr + c_reac + c_ss
            cost_tech = c_react + c_losses  # not constraints per se, but we want to minimize these

            return cost_invest, cost_tech, gs
        
          # return cost_invest, cost_tech, gs
        # Fin Evaluate.3







        # DEPURACIÓN Y PRUEBAS

         # Esta parte conecta las funciones principales en un flujo lógico para validar el modelo o realizar pruebas manuales
         # Permite ejecutar el flujo de cálculo completo fuera del optimizador, útil para verificar que cada componente funciona correctamente
         # No se usa durante la optimización: El optimizador de pymoo ejecutará automáticamente el flujo interno al llamar a _evaluate 


        # Main data

        # Parámetros principales de la red
        nbus = 6
        vslack = 1.0
        dslack = 0.0
        max_iter = 20
        epss = 1e-6

        # Parámetros base del sistema
        Sbase = 100e6  # VA
        f = 50         # Hz
        l = 100        #  distance to shore in km
        p_owf = 2      # p.u, equivalent to 500 MW owf
        q_owf = 0      # p.u, we assume no reactive power is generated at plant


        # Construcción de la red
        Y_bus, p_owf, q_owf, n_cables, u_i, I_rated, S_rtr, Y_l1, Y_l2, Y_l3, Y_l4, Y_l5, A, B, C, y_trserie, y_piserie = build_grid_data(Sbase, f, l, p_owf, q_owf, vol, S_rtr, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1, react2, react3, react4, react5)

        # Cálculo del flujo de potencia
        V_wslack, angle_wslack, curr, p_wslack, q_wslack, solution_found = run_pf(p_owf, q_owf, Y_bus, nbus, vslack, dslack, max_iter, epss, y_trserie, y_piserie)

        #  Cálculo de costos y restricciones
        cost_invest, cost_tech, gs  = compute_costs(p_owf, p_wslack, q_wslack, V_wslack, curr, nbus, n_cables, u_i, I_rated, S_rtr, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, Y_l1, Y_l2, Y_l3, Y_l4, Y_l5, solution_found) 
        # print(cost_output)


        # Preparación para el optimizador
          # Fill F and G <= 0
        out["F"] = np.array([cost_invest, cost_tech], dtype=float)    # Contiene los objetivos del problema de optimización (cost_invest, cost_tech)
        out["G"] = np.array(gs, dtype=float)                          # Contiene las restricciones (gi<=0)
