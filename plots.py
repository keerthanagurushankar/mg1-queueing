import math, random, numpy as np
import matplotlib.pyplot as plt
import lib, simulations, policies as policy
import json, os, logging
plt.rcParams.update({'font.size': 14})

# CONSTANTS

rhos = np.linspace(0.8, 1, 15)[8:-1]


# HELPER FUNCTIONS

def save_sample_path(l1, l2, S1, S2, dname):
    # run an FCFS simulation and save metrics to dname (we only use arrival seqs)
    simulated_MG1 = simulations.MG1([simulations.JobClass(1, l1, S1), 
                                     simulations.JobClass(2, l2, S2)], policy.FCFS)
    simulated_MG1.run()
    simulated_MG1.save_metrics(dname)

def run_2class_simulation(l1, l2, S1, S2, policy, dname=None):
    # run sim: if dname given, use dname/arrival_sequence{1,2}.json to gen sample path
    # return 2 lists of response time samples from sim
    simulated_MG1 = simulations.MG1([simulations.JobClass(1, l1, S1, dname), 
                                     simulations.JobClass(2, l2, S2, dname)], policy)
    simulated_MG1.run()
    T1 = [job['response_time'] for job in simulated_MG1.metrics if job['job_class'] == 1]
    T2 = [job['response_time'] for job in simulated_MG1.metrics if job['job_class'] == 2]
    return T1, T2

def compute_best_cost_for_rho(exp_name, mu1, mu2, C1, C2, p1, rho, policy_fam):
    # return best_cost in policy_fam for arrival_seq(mu1, mu2, rho, p1)
    # compute arrival rates for rho, get or create arrival sequence
    l = rho / (p1 / mu1 + (1-p1) / mu2)
    l1, l2 = round(p1 * l, 3), round((1-p1)* l, 3)
    S1, S2 = lib.exp(mu1), lib.exp(mu2)

    sample_name = f'{exp_name}/MM1-{l1}-{l2}-{mu1}-{mu2}'
    if not os.path.exists(sample_name):
        save_sample_path(l1, l2, S1, S2, sample_name)  

    # run simulation for a given arrival sequence and policy, compute cost
    def compute_cost_for_policy(policy):
        policy = policy(l1, l2) if callable(policy) else policy
        T1, T2 = run_2class_simulation(l1, l2, S1, S2, policy, sample_name)
        ECost =  np.mean(list(map(C1, T1)) + list(map(C2, T2)))
        logging.info(f"Ran simulation for load {round(rho, 3)}, "
                     f"{policy.policy_name} -> {ECost}")
        return ECost

    def cost_for_idx(idx):
        return compute_cost_for_policy(policy_fam[idx])

    # assume cost is unimodal in policy_fam[left_idx: right_idx+1], binary search argmin     
    def search_best_costs(left_idx, right_idx, left_cost=None, right_cost=None):
        if right_idx - left_idx == 0:
            if left_cost:
                return left_idx, left_cost
            if right_cost:
                return right_idx, right_cost
            return left_idx, cost_for_idx(left_idx)

        # compute extreme costs if not provided
        left_cost = cost_for_idx(left_idx) if not left_cost else left_cost
        right_cost = cost_for_idx(right_idx) if not right_cost else right_cost

        curr_min_idx = left_idx if left_cost <= right_cost else right_idx
        curr_min_cost = min(left_cost, right_cost)

        # if range is non-trivial, check midpt
        # mid_idx = (left_idx + right_idx)//2
        mid_idx = int((left_idx/left_cost + right_idx/right_cost)/(1/left_cost+1/right_cost))

        if mid_idx == left_idx or mid_idx == right_idx:
            return curr_min_idx, curr_min_cost
        
        mid_cost = cost_for_idx(mid_idx)

        if mid_cost >= curr_min_cost:
            # min occurs at extreme of range, so return correct extreme
            # may be wrong if min was close to extreme but usually works
            return curr_min_idx, curr_min_cost

        # if mid_cost is lower, continue binary search: check left range for best cost
        left_best_idx, left_best_cost = search_best_costs(left_idx, mid_idx, left_cost, mid_cost)
        if left_best_idx != mid_idx:
            return left_best_idx, left_best_cost
        # if left range best cost was at mid_idx, return right range best cost
        right_best_idx, right_best_cost = search_best_costs(mid_idx, right_idx, mid_cost, right_cost)
        return right_best_idx, right_best_cost

    best_policy, best_cost = search_best_costs(0, len(policy_fam) - 1)
    return best_cost

def compute_best_costs(exp_name, mu1, mu2, c1, c2, p1, policy_fam):
    return [compute_best_cost_for_rho(exp_name, mu1, mu2, c1, c2,
                p1, rho, policy_fam) for rho in rhos]

def time_avg_costs(rhos, costs, p1=0.5):
    mu1, mu2 = 3, 1
    # l * (p1 / mu1 + (1-p1) / mu2) = rho
    ls = rhos / (p1 / mu1 + (1-p1) / mu2)
    return ls * costs

def time_avg_costs2(rhos, costs, p1=0.5):
    mu1, mu2, mu3 = 1, 3, 3
    # l * (p1 / mu1 + (1-p1) / mu2) = rho
    ls = rhos / (1/3 / mu1 + 1/3 / mu2 + 1/3/mu3)
    return ls * costs

def gen_plot(exp_name, costs_by_policy, p1=0.5):
    # given costs_by_policy: dict[str->list[list[float]]] (or read from file)
    costs_fname = f'{exp_name}/costs{p1}.json'
    if costs_by_policy:
        json.dump(costs_by_policy, open(costs_fname, 'w'))
    else:
        costs_by_policy = json.load(open(costs_fname, 'r'))

    #del costs_by_policy['FCFS'] # don't plot FCFS if too far off
    plt.figure()
    plot_style = {'FCFS':('-', 2, 'blue', 0.7),
                  r'gen-$c\mu$': ('--', 2, 'green', 0.7),
                  #'Lookahead*': ('--', 2, 'orange', 0.5),
                  'PPrio':(':', 2, 'purple', 1),
                  #'AccPrio*': ('-.', 2, 'gray', 0.5),                  
                  'Aalto':('-', 2, 'orange', 0.7),
                  'Whittle':('--', 2, 'red', 0.7),                  }

    for policy in plot_style:
        if policy in costs_by_policy:
            costs = costs_by_policy[policy]

            costs = time_avg_costs(rhos, costs, p1)
            
            ls, lw, color, alpha = plot_style[policy]

            if policy == "Whittle":
                policy = "Us"

            if policy == "PPrio":
                policy = "Prio"

            
            #print(rhos) print(costs)
            plt.plot(
                #list(rhos[::2]),list(costs[::2]) ,
                #np.delete(rhos, [1, 4,8, 10]), np.delete(costs, [1, 4,8, 10]),
                np.delete(rhos, [5, 10]),np.delete(costs, [5, 10]),
                #np.delete(rhos, [-4]),np.delete(costs, [-4]),
                #rhos, costs,
                label=policy,linestyle=ls, linewidth=lw, color=color, alpha=alpha)

    #plt.ylim(-0.1e6, 1.45e6) 2deadline drastic
    #plt.ylim(-25, 850) linear drastic
    #plt.ylim(0, 3000) # polynomial balanced
    #plt.ylim(0, 25) # 2 deadline balanced
    #plt.ylim(-300, 20000)
    plt.ylim(0, 3000)
    plt.xlim(0.8, 0.97)
    plt.xlabel('Load')
    plt.ylabel('Time-avg Total Holding Cost')
    plt.legend(loc = "upper left")
    plt.savefig(f'{exp_name}/rhos-vs-costs-{p1}.png')



def gen_plot2(exp_names, costs, p1=0.5):
    """ Generates a plot using the average cost across multiple experiment runs.

    Args:
        exp_names (list[str]): List of experiment names (directories containing cost JSONs).
        rhos (list): List of load values.
        p1 (float): Probability threshold for filename storage.
    """
    
    all_costs_by_policy = []
    
    # Load all experiment data
    for exp_name in exp_names:
        costs_fname = f'{exp_name}/costs{p1}.json'
        if os.path.exists(costs_fname):
            with open(costs_fname, 'r') as f:
                all_costs_by_policy.append(json.load(f))
        else:
            print(f"Warning: {costs_fname} not found.")
    
    if not all_costs_by_policy:
        print("No valid experiment data found. Exiting.")
        return

    # Compute the average cost for each policy
    avg_costs_by_policy = {}
    policies = all_costs_by_policy[0].keys()  # Use first experiment to get policies

    for policy in policies:
        if all(policy in costs for costs in all_costs_by_policy):  # Ensure policy exists in all runs
            avg_costs_by_policy[policy] = np.mean(
                [np.array(costs[policy]) for costs in all_costs_by_policy], axis=0
            )

         
    plot_style = {'FCFS':('-', 2, 'blue', 0.7),
                  r'gen-$c\mu$': ('--', 2, 'green', 0.7),
                  #'Lookahead*': ('--', 2, 'orange', 0.5),
                  'PPrio':(':', 2, 'purple', 1),
                  #'AccPrio*': ('-.', 2, 'gray', 0.5),                  
                  'Aalto':('-', 2, 'orange', 0.7),
                  'Whittle':('--', 2, 'red', 0.7),                  }

    # Generate plot
    plt.figure()

    for policy, (ls, lw, color, alpha) in plot_style.items():
        if policy in avg_costs_by_policy:
            costs = time_avg_costs2(rhos, avg_costs_by_policy[policy])
            
            if policy == "Whittle":
                policy = "Us"

            if policy == "PPrio":
                policy = "Prio"
                
            plt.plot(np.delete(rhos, [2, 4, 8]), np.delete(costs, [2, 4, 8]),
                     label=policy, linestyle=ls, linewidth=lw, color=color, alpha=alpha)

    plt.ylim(0, 10000)
    plt.xlim(0.8, 0.97)
    plt.xlabel('Load')
    plt.ylabel('Time-avg Total Holding Cost')    
    plt.legend(loc="upper left")
    #plt.title(f"Averaged Cost Plot for {', '.join(exp_names)}")
    
    # Save the plot
    plot_filename = f"averaged_plot_{'_'.join(exp_names)}.png"
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
    #plt.close()
