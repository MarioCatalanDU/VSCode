import numpy as np#
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
#  from pymoo.operators.crossover.pntx import TwoPointCrossover
#  from pymoo.operators.mutation.bitflip import BitflipMutation
#  from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


from pymoo.core.mixed import MixedVariableGA
from pymoo.core.variable import Real, Integer, Choice, Binary
from Main.windopti import MixedVariableProblem
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

ff = costac_2
p_owf = 5
x_opf = np.array([3, 2, 1, 1, 0, 1, 0, 0.519, 0.953, 0.0, 0.737, 0.0, 509.72e6])



vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr = x_opf

cost_invest_opf, cost_tech_opf, cost_fullopf = ff(vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr, p_owf)
c_vol, c_curr, c_losses, c_react, cost_tech, c_cab, c_gis, c_tr, c_reac, cost_invest,c_volover, c_volunder, c_ss, average_v = cost_fullopf


costs_opf= [c_losses, c_cab, c_gis, c_tr, c_reac, c_ss]
labels = ['Power losses', 'Cables', 'Switchgears', 'Transformers', 'Reactors', 'Substation']
plt.bar(labels, costs_opf, color='skyblue')
plt.ylabel('Cost [M€]')
plt.title('Breakdown of costs of NSGA-II solution')
plt.xticks(rotation=20, fontsize=18)  # Rotate labels to avoid overlap
#plt.show()

x_nosh = np.array([3, 2, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 509.72e6])
vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr = x_nosh
cost_invest_no, cost_tech_no, cost_full_no = ff(vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr, p_owf)
c_vol, c_curr, c_losses, c_react, cost_tech, c_cab, c_gis, c_tr, c_reac, cost_invest,c_volover, c_volunder, c_ss, average_v = cost_full_no
costs_no= [c_losses, c_cab, c_gis, c_tr, c_reac, c_ss]
labels = ['Power losses', 'Cables', 'Switchgears', 'Transformers', 'Reactors', 'Substation']
plt.bar(labels, costs_no, color='orange')

plt.ylabel('Cost [M€]')
plt.title('Breakdown of costs without compensation')
plt.xticks(rotation=20, fontsize= 18)  # Rotate labels to avoid overlap
plt.show()

labels = ['Power losses', 'Cables', 'Switchgears', 'Transformers', 'Reactors', 'Substation']
cumulative_opf = np.cumsum([0] + costs_opf[:-1])
cumulative_no = np.cumsum([0] + costs_no[:-1])

# Set up the figure
fig, ax = plt.subplots()

# Plot stacked bars for 'With Compensation'
for i, cost in enumerate(costs_opf):
    ax.bar('NSGA-optimal', cost, bottom=cumulative_opf[i], color=plt.cm.Paired(i),label=labels[i])

# Plot stacked bars for 'Without Compensation'
for i, cost in enumerate(costs_no):
    ax.bar('No Compensation', cost, bottom=cumulative_no[i], color=plt.cm.Paired(i), label=labels[i] if i == 0 else "")

# Add labels and title
ax.set_ylabel('Cost [M€]')
ax.set_title('Total Cost Comparison')
plt.xticks(rotation=0)  # Rotate labels to avoid overlap


handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))

legend = plt.legend(unique_labels.values(), unique_labels.keys(), title="Cost Components", bbox_to_anchor=(1.05, 1), loc='upper left')
# Show the plot
plt.tight_layout()
plt.show()

