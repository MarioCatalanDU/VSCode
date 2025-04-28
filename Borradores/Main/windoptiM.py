# windopti_var.py
# ---------------------------------------------------------------------------------
# NÚCLEO DEL PROBLEMA DE OPTIMIZACIÓN CON POTENCIA VARIABLE.
#
# Esta versión es similar a la de windopti (la versión final que te gusta),
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
import pymoo.gradient.toolbox as anp
import time

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice, Binary

# Importamos las funciones de Weibull para generar potencia variable
from Weibull import generate_wind_speeds_weibull, wind_speed_to_power

# ---------------------------------------------------------------------------------
# Clase del Problema de Optimización con Variables Mixtas (Versión Variable)
# ---------------------------------------------------------------------------------
class MixedVariableProblem(ElementwiseProblem):
    
    def __init__(self, tipo_potencia="variable", **kwargs):
        self.tipo_potencia = tipo_potencia

        vars = {                                         
            "react1_bi": Binary(),
            "react2_bi": Binary(),
            "react3_bi": Binary(),
            "react4_bi": Binary(),
            "react5_bi": Binary(),
            "vol_level": Choice(options=["vol2132","vol220"]),
            "n_cables": Integer(bounds=(2, 3)),
            "S_rtr": Real(bounds=(300e6, 900e6)),
            "react1": Real(bounds=(0.0, 1.0)),
            "react2": Real(bounds=(0.0, 1.0)),
            "react3": Real(bounds=(0.0, 1.0)),
            "react4": Real(bounds=(0.0, 1.0)),
            "react5": Real(bounds=(0.0, 1.0))
        }
        super().__init__(vars=vars, n_obj=2, **kwargs)
    
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
        (react1_bi, react2_bi, react3_bi, react4_bi, react5_bi,
         vol, n_cables, S_rtr, react1, react2, react3, react4, react5) = (
            X["react1_bi"], X["react2_bi"], X["react3_bi"],
            X["react4_bi"], X["react5_bi"], X["vol_level"],
            X["n_cables"], X["S_rtr"], X["react1"], X["react2"],
            X["react3"], X["react4"], X["react5"]
        )
        nbus = 6
        vslack = 1.0
        dslack = 0.0
        max_iter = 20
        epss = 1e-6
        Sbase = 100e6
        f = 50
        l = 100
        
        # POTENCIA FIJA, VARIABLE MEDIA O VARIABLE
        # Aquí, en lugar de usar una potencia fija p_owf=5, la calculamos a partir de Weibull.
        # Se genera una distribución de velocidades y se convierte a potencia.
        if self.tipo_potencia == "variable":
            N, k, lambda_ = 10000, 2, 8
            wind_speeds, wind_probs = generate_wind_speeds_weibull(N, k, lambda_, seed=42)
            n_turbines = 36
            power_generated = wind_speed_to_power(wind_speeds, n_turbines=n_turbines)
            p_owf = np.mean(power_generated) / 1e6 / 100.0
            q_owf = 0
            escenarios = [(p_owf, 1.0)]  # media como único escenario

        elif self.tipo_potencia == "variable_extendida":
            N, k, lambda_ = 500, 2, 8
            wind_speeds, wind_probs = generate_wind_speeds_weibull(N, k, lambda_, seed=42)
            n_turbines = 36
            power_generated = wind_speed_to_power(wind_speeds, n_turbines=n_turbines)

            bins = 8
            hist, edges = np.histogram(wind_speeds, bins=bins, weights=wind_probs, density=False)
            probabilidades = hist / hist.sum()
            escenarios = []

            for i in range(bins):
                mask = (wind_speeds >= edges[i]) & (wind_speeds < edges[i+1])
                if np.any(mask):
                    pot_medio = np.mean(power_generated[mask]) / 1e6 / 100.0
                    escenarios.append((pot_medio, probabilidades[i]))

            q_owf = 0

        elif self.tipo_potencia == "fija":
            p_owf = 5
            q_owf = 0
            escenarios = [(p_owf, 1.0)]
        else:
            raise ValueError("El tipo_potencia debe ser 'fija', 'variable', o 'variable_extendida'")
        

        # =====================================
        # === OPTIMIZACIÓN MULTI-ESCENARIO ====
        # =====================================
        coste_inv_total = 0
        coste_tec_total = 0

        for p_owf_i, prob_i in escenarios:
            (Y_bus, _, _, n_cables, u_i, I_rated, S_rtr,
             Y_l1, Y_l2, Y_l3, Y_l4, Y_l5, A, B, C, y_trserie, y_piserie, Y_ref
            ) = self.build_grid_data(Sbase, f, l, p_owf_i, q_owf, vol, S_rtr, n_cables,
                                     react1_bi, react2_bi, react3_bi, react4_bi, react5_bi,
                                     react1, react2, react3, react4, react5)
    
            (V_wslack, angle_wslack, curr, p_wslack, q_wslack, solution_found) = self.run_pf(
                p_owf_i, q_owf, Y_bus, nbus, vslack, dslack,
                max_iter, epss, y_trserie, y_piserie, S_rtr, n_cables, vol
            )
    
            (cost_invest, cost_tech, gs, g1_vol, cost_full) = self.compute_costs(
                p_owf_i, p_wslack, q_wslack, V_wslack, curr,
                nbus, n_cables, u_i, I_rated, S_rtr,
                react1_bi, react2_bi, react3_bi, react4_bi, react5_bi,
                Y_l1, Y_l2, Y_l3, Y_l4, Y_l5, Y_ref, solution_found,
                Sbase, l, A, B, C
            )
    
            coste_inv_total += prob_i * cost_invest
            coste_tec_total += prob_i * cost_tech


        # Aplicamos una penalización en caso de que el costo técnico exceda el umbral deseado
        umbral_tech = 2000.0  # Ajusta este valor según tus expectativas
        if cost_tech > umbral_tech:
             cost_invest = 1e9
             cost_tech = 1e9

        out["F"] = [cost_invest, cost_tech]
        # Puedes definir restricciones adicionales si lo deseas:
        # out["G"] = np.array(gs)