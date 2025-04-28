# Optimizador_var.py
# ---------------------------------------------------------------------------------
# NÚCLEO DEL PROBLEMA DE OPTIMIZACIÓN CON POTENCIA VARIABLE.
#
# Esta versión es similar a la de Optimizador (la versión final que te gusta),
# pero en el método _evaluate se utiliza la potencia variable obtenida de Weibull
# en lugar de una potencia fija (p_owf=5).
#
# Se asume que el módulo Weibull.py provee:
#   - generate_wind_speeds_weibull(N, k, lambda_, seed=None)
#   - wind_speed_to_power(wind_speeds, n_turbines)
#
# Los resultados se convertirán a p.u. de forma que, si la potencia promedio es 500 MW,
# se asigne p_owf = 5.
# ---------------------------------------------------------------------------------

import numpy as np
import cmath
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice, Binary
from Weibull import generate_wind_speeds_weibull, wind_speed_to_power

# ---------------------------------------------------------------------------------
# Clase del Problema de Optimización con Variables Mixtas (Versión Variable)
# ---------------------------------------------------------------------------------
class MixedVariableProblem(ElementwiseProblem):
    def __init__(self,
                 tipo_potencia="variable_extendida",  # Selección del modo de potencia
                 n_scenarios=6,                       # Número de escenarios en multi-escenario
                 N_wind=10000,                        # Muestras para muestreo Weibull
                 k_weib=1.5,                            # Parámetro forma Weibull
                 lam_weib=10,                          # Parámetro escala Weibull
                 seed_wind=42,                        # Semilla para reproducibilidad
                 **kwargs):

        # 1. Almaceno parámetros básicos
        self.tipo_potencia = tipo_potencia
        self.n_scenarios = n_scenarios
        self.Sbase = 100e6  # Base de potencia (100 MVA) para sistema por unidad (p.u.).
        # Esto significa que todas las potencias se normalizan dividiendo por 100 MVA, 
        # de modo que 1 p.u. = 100 MW de potencia activa en el parque. (100 MVA) para convertir a p.u.

        # 2. Defino las variables de diseño para PyMoo
        vars = {
            "react1_bi": Binary(),  # Activo/desactivo compensador 1
            "react2_bi": Binary(),  # Activo/desactivo compensador 2
            "react3_bi": Binary(),  # ... hasta compensador 5
            "react4_bi": Binary(),
            "react5_bi": Binary(),
            "vol_level": Choice(options=["volno132","vol220"]),  # Nivel de tensión
            "n_cables": Integer(bounds=(2, 3)),  # Nº de cables
            "S_rtr": Real(bounds=(300e6, 900e6)), # Tamaño transformador
            # Valores de susceptancia de cada reactivo
            "react1": Real(bounds=(0.0, 1.0)),
            "react2": Real(bounds=(0.0, 1.0)),
            "react3": Real(bounds=(0.0, 1.0)),
            "react4": Real(bounds=(0.0, 1.0)),
            "react5": Real(bounds=(0.0, 1.0))
        }
        # Inicializo el problema con 2 objetivos (coste inversión, coste técnico)
        super().__init__(vars=vars, n_obj=2, **kwargs)

        # 3. PRE-CÁLCULO DE ESCENARIOS DE POTENCIA (solo una vez)
        self.scenarios = []
        
        if tipo_potencia in ("variable", "variable_extendida"):
            # 3.1) Muestreo Weibull de velocidades
            ws, wp = generate_wind_speeds_weibull(N_wind, k_weib, lam_weib, seed=seed_wind)
            # 3.2) Curva de potencia (W)
            P_w = wind_speed_to_power(ws, n_turbines=36)
            # 3.3) Paso a p.u.
            P_pu = P_w / self.Sbase

            if tipo_potencia == "variable":
                p_mean = np.sum(P_pu * wp)
                self.scenarios = [(p_mean, 1.0)]
            else:
                # 3.b Multi-escenario: cuantiles equiespaciados
                cum = np.cumsum(wp)  # función distribución acumulada
                # cuantiles 0.5/n, 1.5/n, ..., (n-0.5)/n
                qs = (np.arange(1, n_scenarios+1) - 0.5) / n_scenarios
                # potencias representativas en esos cuantiles
                p_q = np.interp(qs, cum, P_pu)
                # peso uniforme para cada escenario
                w_q = 1.0 / n_scenarios
                # lista de tuplas (potencia_pu, probabilidad)
                self.scenarios = [(p, w_q) for p in p_q]

        elif tipo_potencia == "fija":
            # Modo P.fija: 5 p.u. = 500 MW sobre base de 100 MVA
            self.scenarios = [(5.0, 1.0)]
        else:
            raise ValueError("tipo_potencia debe ser 'fija', 'variable' o 'variable_extendida'")

    # -------------------------------------------------------------------------
    # Método: build_grid_data (igual que en la versión original)
    # -------------------------------------------------------------------------
    def build_grid_data(self, Sbase, f, l, p_owf, q_owf, vol, S_rtr, n_cables,
                        react1_bi, react2_bi, react3_bi, react4_bi, react5_bi,
                        react1_val, react2_val, react3_val, react4_val, react5_val):
        if vol == "vol132":
            u_i = 132e3
            R = 0.0067
            Cap = 0.25e-6
            L = 0.35e-3
            A = 1.971e6
            B = 0.209e6
            C = 1.66
            I_rated = 825
        elif vol == "vol220":
            u_i = 220e3
            R = 0.0067
            Cap = 0.19e-6
            L = 0.38e-3
            A = 3.181e6
            B = 0.11e6
            C = 1.16
            I_rated = 825
        else:
            u_i = 220e3
            A = 3.181e6; B = 0.11e6; C = 1.16; I_rated = 825
        Y_ref = Sbase / u_i**2
        V_ref = u_i

        # 1.2 TRAFO
        U_rtr = u_i
        P_Cu = 60e3
        P_Fe = 40e3
        u_k = 0.18
        i_o = 0.012
        G_tri = (P_Fe / U_rtr**2)
        B_tri = - (i_o * (S_rtr / U_rtr**2))
        Y_tr = (G_tri + 1j * B_tri) / Y_ref
        R_tr = P_Cu / (3 * (S_rtr / (np.sqrt(3) * u_i))**2)
        X_tr = np.sqrt((u_k * (U_rtr**2 / S_rtr))**2 - R_tr**2)
        Z_tr = R_tr + 1j * X_tr
        Y_trserie = (1 / Z_tr) / Y_ref

        g_tr = np.real(Y_tr)
        b_tr = np.imag(Y_tr)
        r_tr = np.real(Z_tr * Y_ref)
        x_tr = np.imag(Z_tr * Y_ref)
     
        # 1.3 CABLES
        R = 0.0067
        Cap = 0.17e-6
        L_val = 0.40e-3
        Y_val = 1j * (2 * np.pi * f * Cap / 2)
        Z_val = R + 1j * (2 * np.pi * f * L_val)
        theta = l * np.sqrt(Z_val * Y_val)
        Y_pi = n_cables * (Y_val * l / 2 * np.tanh(theta / 2) / (theta / 2)) / Y_ref
        G_pi = np.real(Y_pi)
        B_pi = np.imag(Y_pi)
        Z_piserie = (Z_val * l * np.sinh(theta) / (theta)) / n_cables
        Y_piserie = n_cables * (1 / Z_piserie) / Y_ref 

        # 1.4 COMPENSADORES
        if react1_bi:
            Y_l1 = -1j * react1_val
        else:
            Y_l1 = 0
            react1_bi = 0.0
        if react2_bi:
            Y_l2 = -1j * react2_val
        else:
            Y_l2 = 0
            react2_bi = 0.0
        if react3_bi:
            Y_l3 = -1j * react3_val
        else:
            Y_l3 = 0
            react3_bi = 0.0
        if react4_bi:
            Y_l4 = -1j * react4_val
        else:
            Y_l4 = 0
            react4_bi = 0.0
        if react5_bi:
            Y_l5 = -1j * react5_val
        else:
            Y_l5 = 0
            react5_bi = 0.0

        # 1.5 GRID CONNECTION
        scr = 15
        xrr = 10
        V_grid = 220e3
        zgridm = V_grid**2 / (scr * p_owf * Sbase)
        rgrid = np.sqrt(zgridm**2 / (xrr**2 + 1))
        xgrid = xrr * rgrid
        zgrid = rgrid + 1j * xgrid
        Y_g = (1 / zgrid) / Y_ref

        Y_bus = np.array([
            [Y_trserie + Y_tr + Y_l1, -Y_trserie,                  0,                     0,                     0,      0],
            [-Y_trserie,             Y_piserie + Y_pi + Y_l2 + Y_trserie, -Y_piserie,         0,                     0,      0],
            [0,                      -Y_piserie,                  2 * Y_piserie + 2 * Y_pi + Y_l3, -Y_piserie,        0,      0],
            [0,                       0,                         -Y_piserie,            Y_piserie + Y_pi + Y_l4 + Y_trserie, -Y_trserie, 0],
            [0,                       0,                          0,                   -Y_trserie,            Y_trserie + Y_tr + Y_l5 + Y_g, -Y_g],
            [0,                       0,                          0,                    0,                   -Y_g,             Y_g]
        ], dtype=complex)

        return (Y_bus, p_owf, q_owf, n_cables, u_i, I_rated, S_rtr,
                Y_l1, Y_l2, Y_l3, Y_l4, Y_l5, A, B, C, Y_trserie, Y_piserie, Y_ref)
    
    # -------------------------------------------------------------------------
    # Método: run_pf (Cálculo del flujo de potencia)
    # -------------------------------------------------------------------------
    def run_pf(self, p_owf, q_owf, Y_bus, nbus, V_slack, angle_slack,
               max_iter, eps, y_trserie, y_piserie, S_rtr, n_cables, vol):
        V = np.ones(nbus - 1, dtype=float)
        V_wslack = np.empty(nbus, dtype=float)
        V_wslack[:nbus - 1] = V
        V_wslack[nbus - 1] = V_slack
        angles = np.zeros(nbus - 1, dtype=float)
        angle_wslack = np.empty(nbus, dtype=float)
        angle_wslack[:nbus - 1] = angles
        angle_wslack[nbus - 1] = angle_slack

        x0 = np.concatenate([angles, V])
        x = x0.copy()
        P_obj = np.array([p_owf, 0, 0, 0, 0])
        Q_obj = np.array([q_owf, 0, 0, 0, 0])
        PQ_obj = np.concatenate([P_obj, Q_obj])
        epsilon = 1e10
        k = 0

        while epsilon > eps and k < max_iter:
            k += 1
            x = np.concatenate((angles, V))
            P = np.zeros(nbus - 1)
            Q = np.zeros(nbus - 1)
            for i in range(nbus - 1):
                for j in range(nbus):
                    P[i] += V_wslack[i] * V_wslack[j] * (
                        np.real(Y_bus[i, j]) * np.cos(angle_wslack[i]-angle_wslack[j]) +
                        np.imag(Y_bus[i, j]) * np.sin(angle_wslack[i]-angle_wslack[j])
                    )
            for i in range(nbus - 1):
                for j in range(nbus):
                    Q[i] += V_wslack[i] * V_wslack[j] * (
                        np.real(Y_bus[i, j]) * np.sin(angle_wslack[i]-angle_wslack[j]) -
                        np.imag(Y_bus[i, j]) * np.cos(angle_wslack[i]-angle_wslack[j])
                    )
            deltaPQ = PQ_obj - np.concatenate((P, Q))
            J11 = np.zeros((nbus - 1, nbus - 1))
            J12 = np.zeros((nbus - 1, nbus - 1))
            J21 = np.zeros((nbus - 1, nbus - 1))
            J22 = np.zeros((nbus - 1, nbus - 1))
            for i in range(nbus - 1):
                for j in range(nbus - 1):
                    if j == i:
                        J11[i, j] = - Q[i] - (V[i]**2) * np.imag(Y_bus[i, i])
                    else:
                        J11[i, j] = abs(V[i]) * abs(V[j]) * (
                            np.real(Y_bus[i, j]) * np.sin(angles[i]-angles[j]) -
                            np.imag(Y_bus[i, j]) * np.cos(angles[i]-angles[j])
                        )
            for i in range(nbus - 1):
                for j in range(nbus - 1):
                    if j == i:
                        J12[i, j] = P[i] / abs(V[i]) + np.real(Y_bus[i, i]) * abs(V[i])
                    else:
                        J12[i, j] = abs(V[i]) * (
                            np.real(Y_bus[i, j]) * np.cos(angles[i]-angles[j]) +
                            np.imag(Y_bus[i, j]) * np.sin(angles[i]-angles[j])
                        )
            for i in range(nbus - 1):
                for j in range(nbus - 1):
                    if j == i:
                        J21[i, j] = P[i] - (V[i]**2) * np.real(Y_bus[i, i])
                    else:
                        J21[i, j] = - abs(V[i]) * abs(V[j]) * (
                            np.real(Y_bus[i, j]) * np.cos(angles[i]-angles[j]) +
                            np.imag(Y_bus[i, j]) * np.sin(angles[i]-angles[j])
                        )
            for i in range(nbus - 1):
                for j in range(nbus - 1):
                    if j == i:
                        J22[i, j] = Q[i] / abs(V[i]) - np.imag(Y_bus[i, i]) * abs(V[i])
                    else:
                        J22[i, j] = abs(V[i]) * (
                            np.real(Y_bus[i, j]) * np.sin(angles[i]-angles[j]) -
                            np.imag(Y_bus[i, j]) * np.cos(angles[i]-angles[j])
                        )
            J_top = np.concatenate((J11, J12), axis=1)
            J_bottom = np.concatenate((J21, J22), axis=1)
            J = np.concatenate((J_top, J_bottom), axis=0)
            try:
                delta_x = np.linalg.solve(J, deltaPQ)
            except np.linalg.LinAlgError:
                break
            x_new = x + delta_x  
            angles = x_new[0:nbus - 1]
            V = x_new[nbus - 1:2*(nbus - 1)]
            angle_wslack[:nbus - 1] = angles
            V_wslack[:nbus - 1] = V
            epsilon = max(abs(deltaPQ))
        if k < max_iter:
            solution_found = True
        else:
            solution_found = False
            V_wslack = np.zeros(nbus)
            angle_wslack = np.zeros(nbus)
            curr = np.zeros(nbus - 2)
            p_wslack = np.zeros(nbus)
            q_wslack = np.zeros(nbus)
            return V_wslack, angle_wslack, curr, p_wslack, q_wslack, solution_found

        p_wslack = np.zeros(nbus)
        q_wslack = np.zeros(nbus)
        curr_inj = np.zeros(nbus, dtype="complex")
        curr = np.zeros(nbus - 2)
        for i in range(nbus):
            for j in range(nbus):
                p_wslack[i] += V_wslack[i] * V_wslack[j] * (
                    np.real(Y_bus[i, j]) * np.cos(angle_wslack[i]-angle_wslack[j]) +
                    np.imag(Y_bus[i, j]) * np.sin(angle_wslack[i]-angle_wslack[j])
                )
        for i in range(nbus):
            for j in range(nbus):
                q_wslack[i] += V_wslack[i] * V_wslack[j] * (
                    np.real(Y_bus[i, j]) * np.sin(angle_wslack[i]-angle_wslack[j]) -
                    np.imag(Y_bus[i, j]) * np.cos(angle_wslack[i]-angle_wslack[j])
                )
        for i in range(nbus):
            for j in range(nbus):
                curr_inj[i] += Y_bus[i,j] * cmath.rect(V_wslack[j], angle_wslack[j])
        i_21 = abs((cmath.rect(V[0], angles[0]) - cmath.rect(V[1], angles[1])) * y_trserie)
        i_32 = abs((cmath.rect(V[1], angles[1]) - cmath.rect(V[2], angles[2])) * y_piserie)
        i_43 = abs((cmath.rect(V[2], angles[2]) - cmath.rect(V[3], angles[3])) * y_piserie)
        i_54 = abs((cmath.rect(V[3], angles[3]) - cmath.rect(V[4], angles[4])) * y_trserie)
        curr = np.array([i_21, i_32, i_43, i_54])
        return V_wslack, angle_wslack, curr, p_wslack, q_wslack, solution_found

    # -------------------------------------------------------------------------
    # Método: compute_costs (Cálculo de Costos y Restricciones)
    # -------------------------------------------------------------------------
    def compute_costs(self, p_owf, p_wslack, q_wslack, V, curr, nbus, n_cables,
                      u_i, I_rated, S_rtr, react1_bi, react2_bi, react3_bi,
                      react4_bi, react5_bi, Y_l1, Y_l2, Y_l3, Y_l4, Y_l5,
                      Y_ref, solution_found, Sbase, l, A, B, C):
        if p_wslack is None:
            p_wslack = np.zeros(nbus)
        if q_wslack is None:
            q_wslack = np.zeros(nbus)
        if V is None:
            V = np.ones(nbus)
        p_lossac = Sbase * (p_owf + p_wslack[5]) * 1e-6  # MW

        Sncab = np.sqrt(3) * u_i * I_rated
        eur_sek = 0.09
        c_cab = n_cables * (A + B * np.exp(C * Sncab / 1e8)) * l * eur_sek / 1e6
        c_gis = (0.0017 * u_i * 1e-3 + 0.0231)
        c_ss = (2.534 + 0.0887 * p_owf * 100)
        t_owf = 25
        c_ey = 100
        c_losses = (8760 * t_owf * c_ey * p_lossac) / 1e6
        c_tr = (0.0427 * (S_rtr * 1e-6)**0.7513)
        fact = 1
        k_on = 0.01049 * fact
        k_mid = 0.01576 * fact
        k_off = 0.01576 * fact
        p_on = 0.8312
        p_mid = 12.44
        p_off = 1.244

        if react1_bi:
            c_r1 = k_off * (abs(Y_l1) * Y_ref * (V[0])**2) + p_off
        else:
            c_r1 = 0
        if react2_bi:
            c_r2 = k_off * (abs(Y_l2) * Y_ref * (V[1])**2) + p_off
        else:
            c_r2 = 0
        if react3_bi:
            c_r3 = k_mid * (abs(Y_l3) * Y_ref * (V[2])**2) + p_mid
        else:
            c_r3 = 0
        if react4_bi:
            c_r4 = k_on * (abs(Y_l4) * Y_ref * (V[3])**2) + p_on
        else:
            c_r4 = 0
        if react5_bi:
            c_r5 = k_on * (abs(Y_l5) * Y_ref * (V[4])**2) + p_on
        else:
            c_r5 = 0

        c_reac = (c_r1 + c_r2 + c_r3 + c_r4 + c_r5) * 1
        penalty = 100
        c_react = 0
        if q_wslack[nbus-1] != 0:
            c_react = abs(q_wslack[nbus-1]) * penalty

        c_vol = 0
        for i in range(nbus - 1):
            if V[i] > 1.1:
                c_vol += (V[i] - 1.1) * penalty
            elif V[i] < 0.9:
                c_vol += (0.9 - V[i]) * penalty  
        i_max_tr = S_rtr / Sbase
        c_curr = 0
        if curr is not None and len(curr) > 0:
            if abs(curr[0]) > 1.1 * i_max_tr:
                c_curr += (abs(curr[0]) - i_max_tr) * penalty
            if abs(curr[-1]) > 1.1 * i_max_tr:
                c_curr += (abs(curr[-1]) - i_max_tr) * penalty

        cost_tech = c_vol + c_curr + c_react + c_losses
        cost_invest = c_cab + c_gis + c_tr + c_reac + c_ss

        cost_full = [c_vol, c_curr, c_losses, c_react, cost_tech,
                     c_cab, c_gis, c_tr, c_reac, c_ss, cost_invest]

        gs = [ (V[i]-1.1) if V[i]>1.1 else (0.9-V[i]) for i in range(nbus-1) ]
        g1_vol = c_vol
        

        return cost_invest, cost_tech, gs, g1_vol, cost_full

    # -------------------------------------------------------------------------
    # Método _evaluate: Conecta los métodos anteriores, usando potencia variable.
    # -------------------------------------------------------------------------
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluación del problema: recorre cada escenario de potencia,
        hace build_grid_data → run_pf → compute_costs,
        y suma los costes ponderados.
        """
        # 1) Extraigo valores de diseño
        react1_bi, react2_bi, react3_bi = X["react1_bi"], X["react2_bi"], X["react3_bi"]
        react4_bi, react5_bi = X["react4_bi"], X["react5_bi"]
        vol = X["vol_level"]
        n_cables = X["n_cables"]
        S_rtr = X["S_rtr"]
        react1_val, react2_val = X["react1"], X["react2"]
        react3_val, react4_val, react5_val = X["react3"], X["react4"], X["react5"]

        # 2) Parámetros fijos del sistema
        nbus = 6           # Nº nodos
        v_slack = 1.0      # Magnitud nodo slack
        d_slack = 0.0      # Ángulo slack
        max_iter = 20      # Iteraciones flujo
        epss = 1e-6        # Tolerancia flujo
        f = 50             # Frecuencia (Hz)
        l = 100            # Longitud típica (km)

        # 3) Inicializo acumuladores de coste
        coste_inv_total = 0.0
        coste_tec_total = 0.0

        # 4) Bucle multi-escenario
        for p_owf_i, prob_i in self.scenarios:
            # 4.a) Construcción de datos de red para p_owf_i
            (Y_bus, _, _, n_cab, u_i, I_r, S_r,
             Y_l1, Y_l2, Y_l3, Y_l4, Y_l5,
             A, B, C, y_trserie, y_piserie, Y_ref) = \
                self.build_grid_data(self.Sbase, f, l,
                                     p_owf_i, 0,
                                     vol, S_rtr, n_cables,
                                     react1_bi, react2_bi,
                                     react3_bi, react4_bi,
                                     react5_bi,
                                     react1_val, react2_val,
                                     react3_val, react4_val,
                                     react5_val)
            # 4.b) Resolución flujo de potencia
            V_w, ang_w, curr, p_ws, q_ws, sol = \
                self.run_pf(p_owf_i, 0,
                            Y_bus, nbus,
                            v_slack, d_slack,
                            max_iter, epss,
                            y_trserie, y_piserie,
                            S_r, n_cab, vol)
            # 4.c) Cálculo de costes para este escenario
            c_inv, c_tec, *_ = self.compute_costs(
                p_owf_i, p_ws, q_ws,
                V_w, curr,
                nbus, n_cab,
                u_i, I_r, S_r,
                react1_bi, react2_bi,
                react3_bi, react4_bi,
                react5_bi,
                Y_l1, Y_l2, Y_l3,
                Y_l4, Y_l5,
                Y_ref, sol,
                self.Sbase, l,
                A, B, C)

            # 4.d) Acumulo coste ponderado por probabilidad
            coste_inv_total += prob_i * c_inv
            coste_tec_total += prob_i * c_tec

        # 5) Penalización si el coste técnico excede un umbral
        if coste_tec_total > 2000.0:
            coste_inv_total = coste_tec_total = 1e9

        # 6) Salida a PyMoo
        out["F"] = [coste_inv_total, coste_tec_total]


def evaluate_solution(problem, x, disable_reactors=False):
    if isinstance(x, dict):
        vars_ = x.copy()
    else:
        vars_ = {name: x[i] for i, name in enumerate(problem.names)}

    # Parámetros internos coherentes con _evaluate
    Sbase = problem.Sbase
    F = 50
    l = 100  # Valor fijo de longitud de cable, como en _evaluate

    if disable_reactors:
        for i in range(1, 6):
            vars_[f"react{i}_bi"] = 0
            vars_[f"react{i}"] = 0.0

    costes_totales = {k: 0.0 for k in [
        "vol_penal", "curr_penal", "losses", "react_penal", "coste_tec",
        "c_cables", "c_gis", "c_trafo", "c_reactores", "c_subest", "coste_inv"]}

    for escenario in problem.scenarios:
        if len(escenario) == 3:
            # (potencia, reactiva, probabilidad)
            p_owf, q_owf, prob = escenario
        elif len(escenario) == 2:
            # (potencia, probabilidad)
            p_owf, prob = escenario
            q_owf = 0.0
        else:
            raise ValueError(f"Formato inesperado en escenario: {escenario}")

        if p_owf < 0.1:
            p_owf = 0.1

        data = problem.build_grid_data(
            Sbase, F, l, p_owf, q_owf,
            vars_["vol_level"], vars_["S_rtr"], vars_["n_cables"],
            vars_["react1_bi"], vars_["react2_bi"], vars_["react3_bi"],
            vars_["react4_bi"], vars_["react5_bi"],
            vars_["react1"], vars_["react2"], vars_["react3"],
            vars_["react4"], vars_["react5"])

        Y_bus, _, _, n_cables, u_i, I_rated, S_rtr, Y_l1, Y_l2, Y_l3, Y_l4, Y_l5, A, B, C, y_trserie, y_piserie, Y_ref = data

        V_w, ang_w, curr, p_ws, q_ws, sol = problem.run_pf(
            p_owf, q_owf, Y_bus, 6, 1.0, 0.0, 20, 1e-6,
            y_trserie, y_piserie,
            vars_["S_rtr"], vars_["n_cables"], vars_["vol_level"])

        cost_invest, cost_tech, _, _, full_costs = problem.compute_costs(
            p_owf, p_ws, q_ws, V_w, curr, 6, n_cables, u_i, I_rated, S_rtr,
            vars_["react1_bi"], vars_["react2_bi"], vars_["react3_bi"],
            vars_["react4_bi"], vars_["react5_bi"],
            Y_l1, Y_l2, Y_l3, Y_l4, Y_l5, Y_ref, sol,
            Sbase, l, A, B, C)

        for i, k in enumerate(costes_totales):
            costes_totales[k] += full_costs[i] * prob

    return costes_totales



