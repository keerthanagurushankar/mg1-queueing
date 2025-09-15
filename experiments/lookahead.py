import math, random, numpy as np, json, os
import matplotlib.pyplot as plt
import scipy.optimize as opt
import policies as policy, lib
from . import utils
import logging
import uuid

# CONSTANTS
rhos = np.linspace(0.75, 1, 10)[:-1]

def compute_lookahead_amount(mu1, mu2, c1_fn, c2, X_rate=0, max_age=20):
    # find t s.t. mu1 E[c1(t + X)] - mu2 c2 = 0
    X = [random.expovariate(X_rate) for _ in range(10**5)]
    overtake_cond = lambda t: mu1 * np.mean([c1_fn(t+Xi) for Xi in X]) - mu2 * c2
    if overtake_cond(0) <= 0 and overtake_cond(max_age) <= 0:
        logging.warning("Lookahead doesn't cross, check max_age")
        return float('inf')
    elif overtake_cond(0) >= 0 and overtake_cond(max_age) >= 0:        
        return 0
    else:
        lookahead_amount = opt.brentq(overtake_cond, 0, max_age)
        return lookahead_amount
    
def gen_lookahead_amount_plot(exp_name, policies, mu1, mu2, p1):
    lookahead = policies["Whittle"][0]
    def lookahead_amount(rho):
        l = rho / (p1 / mu1 + (1-p1) / mu2)
        l1, l2 = p1 * l, (1-p1)* l
        return lookahead([l1, l2]).lookahead_amount
    
    plt.figure()
    plt.plot(rhos, len(rhos)*[policies[r"gen-$c\mu$"][0].lookahead_amount],
             label=r"gen-$c\mu$",linestyle='-', linewidth=2, color='green', alpha=0.7)
    plt.plot(rhos, len(rhos)*[policies["Aalto"][0].lookahead_amount],
             label="Aalto",linestyle='-', linewidth=2, color='orange', alpha=0.7)    
    plt.plot(rhos, [lookahead_amount(rho) for rho in rhos], label="Us",
             linestyle='-', linewidth=2, color='red', alpha=0.7)
    plt.plot(rhos, len(rhos)*[0],
             label="Prio",linestyle='-', linewidth=2, color='purple', alpha=0.7)    
    plt.xlabel('Load')
    plt.ylabel('Overtake age')
    #plt.legend()
    plt.legend(loc = "lower left")   

def run_lookahead_exp(exp_name, mu1, mu2, c1_fn, C1_fn, c2, p1=0.5, max_age=20):
    C2_fn = lambda t : c2 * t
    
    aalto_amount = compute_lookahead_amount(mu1, mu2, c1_fn, c2, mu1, max_age)
    aalto = policy.Lookahead(aalto_amount)

    gen_cmu_amount = compute_lookahead_amount(mu1, mu2, c1_fn, c2, float('inf'), max_age)
    gen_cmu = policy.Lookahead(gen_cmu_amount)

    def lookahead(arrival_rates):
        alpha_star = compute_lookahead_amount(mu1, mu2, c1_fn, c2, 
                                              mu1-arrival_rates[0], max_age)
        return policy.Lookahead(alpha_star)

    policies = { "FCFS": [policy.FCFS],
                 #"AccPrio*": accprios,
                 "Whittle": [lookahead],
                 "Aalto": [aalto],
                 r'gen-$c\mu$' : [gen_cmu],
                 'PPrio': [policy.PPrio12, policy.PPrio21]}
    
    costs_by_policy = {name: utils.compute_best_costs(exp_name, rhos, [mu1, mu2], [C1_fn, C2_fn],
                        [p1, 1-p1], policy_fam) for name, policy_fam in policies.items()}
    utils.gen_plot(exp_name, rhos, costs_by_policy)

    #gen_lookahead_amount_plot(exp_name, policies, mu1, mu2)
    return costs_by_policy

def run_lookahead_exp_error_bars(exp_name, mu1, mu2, c1_fn, C1_fn, c2, p1=0.5, max_age=20, num_runs=3, load_from_file=False):

    if load_from_file:
        dname = f'{exp_name}/runs'
        costs_by_policy = json.load(open(dname + '/costs_by_policy.json', 'r'))
    else:
        dname = f'{exp_name}/runs'
        os.makedirs(dname, exist_ok=True)  # Ensure directory exists
        costs_file = dname + '/costs_by_policy.json'

        # Load previous costs if file exists
        if os.path.exists(costs_file):
            with open(costs_file, 'r') as f:
                old_costs_by_policy = json.load(f)
        else:
            old_costs_by_policy = []

        new_costs_by_policy = []
        for run in range(num_runs):
            run_id = uuid.uuid4().hex[:8]  # Generate a random 8-character ID
            run_exp_name = f"{exp_name}/run{run_id}"
            costs = run_lookahead_exp(run_exp_name, mu1, mu2, c1_fn, C1_fn, c2, p1, max_age)
            new_costs_by_policy.append(costs)

        # Combine old and new costs
        costs_by_policy = old_costs_by_policy + new_costs_by_policy

        # Save combined costs
        with open(costs_file, 'w') as f:
            json.dump(costs_by_policy, f)
    
    policies = ["FCFS", "Whittle", "Aalto", r'gen-$c\mu$', "PPrio"] #costs_by_policy[0].keys()
    baseline_policy = 'Whittle'
    cost_ratios = {policy: np.array([costs_by_policy[run][policy] for run in range(num_runs)]) /
                           np.array([costs_by_policy[run][baseline_policy] for run in range(num_runs)])
                             for policy in policies}
    # costs_by_policy[run][policy] : rho -> ECost[run[rho]]
    return cost_ratios

if __name__ == "__main__":

    c1_fn = lambda t : 10 if t > 10 else 0
    C1_fn = lambda t : 10 * max(0, t-10)
    #run_lookahead_exp('lookahead2', 3, 1, c1_fn, C1_fn, 1, 0.9)
    #utils.gen_plot('lookahead2', rhos, None)
    #utils.plot_ratios('lookahead1', rhos, None, base_policy='Whittle')
    #cost_ratios = run_lookahead_exp_error_bars('lookahead_fig7-3', 3, 1, c1_fn, C1_fn, 1, 0.9, num_runs=10, load_from_file=False)

    c1_fn = lambda t : t**2
    C1_fn = lambda t: t**3 /3
    cost_ratios = run_lookahead_exp_error_bars('lookahead_fig6-4', 1, 3, c1_fn, C1_fn, 30, 0.75, num_runs=10, load_from_file=False)
    print(cost_ratios)
    plt.show()