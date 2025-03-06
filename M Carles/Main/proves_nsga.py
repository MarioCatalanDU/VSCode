import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
#  from pymoo.operators.crossover.pntx import TwoPointCrossover
#  from pymoo.operators.mutation.bitflip import BitflipMutation
#  from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


from pymoo.core.mixed import MixedVariableGA
from pymoo.core.variable import Real, Integer, Choice, Binary
from windopti import MixedVariableProblem
#from windopti_withcstr import MixedVariableProblem2
#from windopti_fum import MixedVariableProblem3
#from windopti_constraints import MixedVariableProblem_constraints
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.algorithms.moo.nsga2 import RankAndCrowding
from pymoo.constraints.as_penalty import ConstraintsAsPenalty
from pymoo.decomposition.asf import ASF
import matplotlib.pyplot as plt
from pymoo.core.evaluator import Evaluator
from pymoo.core.individual import Individual
import time
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.mixed import MixedVariableSampling 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from costac_2 import costac_2

problem = MixedVariableProblem()
#problem = MixedVariableProblem2()
#problem = MixedVariableProblem3()


#problem = MixedVariableProblem_constraints()

algorithm = MixedVariableGA(pop_size = 150, sampling=MixedVariableSampling() ,survival=RankAndCrowding(crowding_func="pcd"))
start_time = time.time()
res = minimize(problem,
               algorithm,
               termination=('n_gen', 6),
               seed=1,
               verbose=True,
               save_history=True)




"""
res = minimize(ConstraintsAsPenalty(problem, penalty=100.0),
               algorithm,
               seed=1,
               verbose=False)

res = Evaluator().eval(problem, Individual(X=res.X))
"""
"""
# Plot the convergence using the history object
n_evals = np.array([e.evaluator.n_eval for e in res.history])
opt = np.array([e.opt[0].F for e in res.history])
plt.title("Convergence")
plt.plot(n_evals, opt, "--")
plt.yscale("log")
plt.show()
"""

#Choice of decicision point (we need weights for each objective)
weights = np.array([0.5, 0.5])
decomp = ASF()
I = decomp(res.F, weights).argmin()

#print(res.F)
print("Best solution found: \nX = %s\nF = %s\nC = %s" % (res.X, res.F, res.CV))
#print(res.history)

#print(res.H)
end_time = time.time()
execution_time = end_time - start_time
print("Execution time NSGA: ", execution_time,"s")
#min_row_index = np.argmin(np.min(res.F, axis=1))
#min_row = res.F[min_row_index]
#print("min row =", min_row)
#print(res.X)
#print(res.CV)
# print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
"""
X, F = res.opt.get("X", "F")
hist = res.history
print("lenHisory =",len(hist))
n_evals = np.array([e.evaluator.n_eval for e in res.history])
opt = np.array([e.opt[0].F for e in res.history])

plt.title("Convergence")
plt.plot(n_evals, opt, "--")
plt.yscale("log")
plt.show()
"""

plot = Scatter()

#plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="black")
# plot best point with weights aproach
plot.add(res.F[I], color="red", s=50)
print("Best solution found weigthed: \nX = %s\nF = %s" % (res.X[I], res.F[I]))
print(res.F.shape)
plot.show()

start_time2 = time.time()
trials = 700
ff = costac_2
random_check = np.zeros((trials,6))
d = 13
num_int = 7
#lb = np.array([3, 2, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 450e6])  # Lower bound
#ub = np.array([3, 3, 1, 1, 1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1000e6])  # Upper bound

lb = np.array([3, 2, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 500e6])  # Lower bound
ub = np.array([3, 3, 1, 1, 1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1000e6])  # Upper bound

p_owf = 5
steps= 20
p_owflist = np.linspace(1, p_owf, trials)
x_history = np.zeros((trials, d))


xnsga = np.array([3, 2, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 700e6])
for i in range(trials):
        
        x0 = np.zeros(d) # Initial guess
        x0[0:num_int] = np.round(np.random.rand(num_int)*(ub[0:num_int]-lb[0:num_int]) + lb[0:num_int]) # Random initial guess (integer)
        x0[num_int:d] = np.random.rand(d-num_int)*(ub[num_int:d]-lb[num_int:d]) + lb[num_int:d] # Random initial guess (continuous)
        x_history[i,:] = x0
        h = np.copy(x0[0:num_int]).astype(int)
        x_history[i,:] = x0
        vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr = x0
       
        cost_invest, cost_tech, cost_full = ff(vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr, p_owf)
        #result = ff(h[0], h[1], h[2], h[3] ,h[4] , h[5] ,h[6], x0[7],x0[8],x0[9],x0[10],x0[11],x0[12])
        random_check[i,:] = [cost_invest, cost_tech, cost_full[10], cost_full[2], cost_full[3], cost_full[11]]
        cost_losses_no = random_check[:,3]


# Find the index of the row with the smallest sum
min_sum_row_index = np.argmin(np.sum(random_check, axis=1))

print(random_check)
# Print the row
print(random_check[min_sum_row_index])
print(x_history[min_sum_row_index,:])

end_time2 = time.time()
execution_time2 = end_time2 - start_time2
print("Execution time random: ", execution_time2,"s")
plt.scatter(random_check[:,0], random_check[:,1], facecolor="none", edgecolor="black", label='Random search')
plt.scatter(res.F[:,0], res.F[:,1], facecolor="none", edgecolor="red",label='NSGA-II Pareto Front')
plt.xlabel("Investment cost [M€]")
plt.ylabel("Technical cost [M€]")
plt.title("Set of solutions comparing NSGA and random search")

plt.show()

"""
ff = costac_2
p_owf = 5
x_opf = np.array([3, 2, 1, 1, 0, 1, 0, 0.519, 0.953, 0.0, 0.737, 0.0, 509.72e6])
x_nosh = np.array([3, 2, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 509.72e6])
vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr = x_opf
cost_invest_opf, cost_tech_opf, cost_fullopf = ff(vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr, p_owf)
c_vol, c_curr, c_losses, c_react, cost_tech, c_cab, c_gis, c_tr, c_reac, cost_invest,c_volover, c_volunder, c_ss = cost_fullopf
plt.scatter(res.F[:,0], res.F[:,1], facecolor="none", edgecolor="black",label='NSGA-II Pareto Front')
plt.scatter(cost_invest_opf, cost_tech_opf, color='green',s=100, label='OPF solution')
plt.scatter(res.F[I,0], res.F[I,1], color='red',s= 80, label='NSGA-II decision point')
plt.xlabel("Investment cost [M€]")
plt.ylabel("Technical cost [M€]")
plt.title("Set of solutions comparing NSGA and OPF")
plt.legend()
plt.show()

costs= [c_losses, c_cab, c_gis, c_tr, c_reac, c_ss]
labels = ['Power losses', 'Cables', 'GIS', 'Transformers', 'Reactive power compensation', 'Substation']
plt.bar(labels, costs, color='skyblue')
plt.xlabel('Cost Components')
plt.ylabel('Cost [M€]')
plt.title('Breakdown of Full OPF Cost')
plt.xticks(rotation=45)  # Rotate labels to avoid overlap
plt.show()
"""