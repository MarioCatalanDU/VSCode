import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import costac_2



trials = 100
ff = costac_2.costac_2
random_check = np.zeros((trials,6))
random_checkpf = np.zeros((trials,6))
avg_list = np.zeros(trials)
avg_listpf = np.zeros(trials)
d = 13
num_int = 7
#lb = np.array([3, 2, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 450e6])  # Lower bound
#ub = np.array([3, 3, 1, 1, 1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1000e6])  # Upper bound

lb = np.array([3, 2, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 450e6])  # Lower bound
ub = np.array([3, 2, 1, 1, 1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1000e6])  # Upper bound

p_owf = 5
p_owflist = np.linspace(2, p_owf, trials)
p_owflistpf = np.linspace(2, p_owf, trials)
x_history = np.zeros((trials, d))

x0 = np.array([3, 2, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 509.72e6])
x_opf = np.array([3, 2, 1, 1, 0, 1, 0, 0.519, 0.953, 0.0, 0.737, 0.0, 509.72e6])
#xnsga = np.array([3, 2, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 700e6])
for i in range(trials):
        
        
        x_history[i,:] = x0
        vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr = x0
        p_owf = p_owflist[i]
        
        cost_invest, cost_tech, cost_full = ff(vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr, p_owf)
        #result = ff(h[0], h[1], h[2], h[3] ,h[4] , h[5] ,h[6], x0[7],x0[8],x0[9],x0[10],x0[11],x0[12])
        random_check[i,:] = [cost_invest, cost_tech, cost_full[10], cost_full[2], cost_full[3], cost_full[11]]
        average_v = cost_full[13]
        avg_list[i] = average_v
        cost_losses_no = random_check[:,3]

for i in range(trials):
        
        
        x_history[i,:] = x_opf
        vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr = x_opf
        p_owf = p_owflistpf[i]

        cost_investpf, cost_techpf, cost_fullpf = ff(vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr, p_owf)
        #print(cost_full[2])
        #result = ff(h[0], h[1], h[2], h[3] ,h[4] , h[5] ,h[6], x0[7],x0[8],x0[9],x0[10],x0[11],x0[12])
        random_checkpf[i,:] = [cost_invest, cost_tech, cost_fullpf[10], cost_fullpf[2], cost_fullpf[3], cost_full[11]]
        average_vpf = cost_fullpf[13]
        avg_listpf[i] = average_vpf
        cost_losses_pf = random_checkpf[:,3]
        
plt.plot(p_owflist, avg_list, label='Average Node Voltage no compensation')
plt.plot(p_owflist, avg_listpf, label='Average Node Voltage optimal compensation')
plt.xlabel('Power Injection [p.u]')
plt.ylabel('Average Node Voltage [p.u]')
plt.axhline(y=1.1, color='r', linestyle='--', label='Technical Constraint at 1.1 p.u.')
plt.title('Average Node Voltage at different wind conditions for an 500 MW OWPP')
plt.legend()
plt.show()




plt.plot(p_owflist, cost_losses_no, label='Power losses with no compensation')
plt.plot(p_owflist, cost_losses_pf, label='Power losses with optimal compensation')
plt.xlabel('Power Injection [p.u]')
plt.ylabel('Power losses [M€/year]')
plt.title('Power losses at different wind conditions for a 500 MW OWPP')
plt.legend()
plt.show()


"""
plt.plot(p_owflist, avg_list, label='Average Node Voltage')



for i in range(trials):
       
        x0 = np.array([3,2,1,1,1,1,1,0.4,0.0,0.8,0.0,0.4,600e6])
        x_history[i,:] = x0
        vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr = x0
        p_owf = p_owflist[i]
        
        cost_invest, cost_tech, cost_full = ff(vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr, p_owf)
        #result = ff(h[0], h[1], h[2], h[3] ,h[4] , h[5] ,h[6], x0[7],x0[8],x0[9],x0[10],x0[11],x0[12])
        random_check[i,:] = [cost_invest, cost_tech, cost_full[10], cost_full[2], cost_full[3], cost_full[11]]

        cost_losses_yes = random_check[:,3]
    
xnsga = np.array([3, 2, 1, 0, 0, 0, 1, 0.702, 0.0, 0.0, 0.0, 0.839, 666.31e6])
vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr = xnsga
cost_invest_nsga, cost_tech_nsga, cost_fullnsga = ff(vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr, p_owf)        

xopf = np.array([3, 2, 1, 0, 0, 0, 1, 0.1711, 0.0, 0.0, 0.0, 0.4775, 666.31e6])
vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr = xopf
cost_invest_opf, cost_tech_opf, cost_fullopf = ff(vol, n_cables, react1_bi, react2_bi, react3_bi, react4_bi, react5_bi, react1_val, react2_val, react3_val,react4_val, react5_val, S_rtr, p_owf)        
 
print(cost_invest_nsga, cost_tech_nsga)
print(cost_invest_opf, cost_tech_opf)
# print(random_check)
plt.scatter(random_check[:,0], random_check[:,1], facecolor="none", edgecolor="black")
plt.scatter(cost_invest_nsga, cost_tech_nsga, color='red', s=50)
plt.scatter(cost_invest_opf, cost_tech_opf, color='green')
plt.ylim(0,1500)
plt.xlim(0,500)
# Find the index of the row with the smallest sum
min_sum_row_index = np.argmin(np.sum(random_check, axis=1))

#print(random_check)
# Print the row
#print(random_check[min_sum_row_index])
#print(x_history[min_sum_row_index,:])

plt.show()
 
#normalized_cost_volover = random_check[:,2] / np.max(random_check[:,2]) + 1e-8
#normalized_cost_losses = random_check[:,3] / np.max(random_check[:,3]) + 1e-8
#normalized_cost_react = random_check[:,4] / np.max(random_check[:,4]) + 1e-8
#normalized_cost_volunder = random_check[:,5] / np.max(random_check[:,5]) + 1e-8


cost_losses = random_check[:,3]
plt.plot(p_owflist, normalized_cost_volover, label='Overvoltage Cost')
plt.plot(p_owflist, normalized_cost_losses, label='Losses Cost')
plt.plot(p_owflist, normalized_cost_react, label='Reactive Power grid Cost')
plt.plot(p_owflist, normalized_cost_volunder, label='Undervoltage Cost')
#plt.plot(p_owflist, cost_losses_no, label='Power losses with no compensation')
plt.title('Evolution of  Costs for Different Power Injections')
plt.xlabel('Power Injection [p.u]')
plt.ylabel('Normalized Costs')
plt.legend()
plt.show()


plt.plot(p_owflist, cost_losses_yes, label='Power losses with compensation')
plt.xlabel('Power Injection [p.u]')
#plt.ylabel('Normalized Costs')
plt.ylabel('Power losses [M€/year]')

#yticks = np.arange(min(cost_losses_yes), max(cost_losses_yes), step=(max(cost_losses_yes)-min(cost_losses_yes))/10)
#plt.yticks(yticks)
plt.grid(axis='y', linestyle='--')

plt.title('Power losses at different wind conditions for a 500 MW wind farm', weight='bold')
plt.legend()

plt.show()

# np.savetxt("random_check.csv", random_check, delimiter=",")
"""