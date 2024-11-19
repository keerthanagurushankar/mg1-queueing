import math, random, numpy as np
import matplotlib.pyplot as plt
import lib, simulations, policies as policy
import json, os, logging
from scipy import integrate

# CONSTANTS

rhos = np.linspace(0.5, 1, 5)[:-1]

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
    # return best_cost, best_policy in policy_fam for arrival_seq(mu1, mu2, rho, p1)
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
        logging.info(f"Ran computations for load {round(rho, 3)}, "
                     f"{policy.policy_name} -> {ECost}")
        return ECost

    def cost_for_idx(idx):
        return compute_cost_for_policy(policy_fam[idx])

    # assume cost is unimodal in [left_idx, right_idx], binary search argmin     
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

        if right_idx - left_idx <= 1:
            return curr_min_idx, curr_min_cost

        mid_idx = (left_idx + right_idx)//2
        mid_cost = cost_for_idx(mid_idx)

        if mid_cost >= curr_min_cost:
            # min occurs at extreme of range, so return correct extreme
            return curr_min_idx, curr_min_cost

        # check left range for best cost
        left_best_idx, left_best_cost = search_best_costs(left_idx, mid_idx, left_cost, mid_cost)
        if left_best_idx != mid_idx:
            return left_best_idx, left_best_cost
        # if left range extreme was mid_idx, return right range best cost
        right_best_idx, right_best_cost = search_best_costs(mid_idx, right_idx, mid_cost, right_cost)
        return right_best_idx, right_best_cost

    best_policy, best_cost = search_best_costs(0, len(policy_fam) - 1)
    return best_cost

def compute_best_costs(exp_name, mu1, mu2, c1, c2, p1, policy_fam):
    return [compute_best_cost_for_rho(exp_name, mu1, mu2, c1, c2,
                p1, rho, policy_fam) for rho in rhos]

def gen_plot(exp_name, costs_by_policy, p1=0.25):
    # given costs_by_policy: dict[str->list[float]] (or read from file)
    costs_fname = f'{exp_name}/costs{p1}.json'
    if costs_by_policy:
        json.dump(costs_by_policy, open(costs_fname, 'w'))
    else:
        costs_by_policy = json.load(open(costs_fname, 'r'))

    # del costs_by_policy['FCFS'] # don't plot FCFS if too far off
    plt.figure()
    plot_style = {'FCFS':('-', 1, 'blue'),
                  'AccPrio*': ('-.', 2, 'gray'),
                  r'gen-$c\mu$': ('-', 1, 'green'),
                  'Lookahead*': ('--', 2, 'orange'),
                  'Whittle':('-', 1, 'red'),
                  'PPrio':(':', 2, 'magenta')}

    for policy, cost in costs_by_policy.items():
        ls, lw, color = plot_style[policy]
        plt.plot(rhos, cost, label=policy, linestyle=ls, linewidth=lw, color=color)
    
    plt.xlabel('Load')
    plt.ylabel('Cost')
    plt.legend()
    plt.savefig(f'{exp_name}/rhos-vs-costs-{p1}.png')

# EXPERIMENT FUNCTIONS

def run_1deadline_exp(mu1, mu2, c1, c2, d1, p1=0.25):
    # instantaneous c1(t) = c1 * is(t > d1), c2(t) = c2     
    exp_name = '1deadline_exp'
    C1_fn = lambda t : c1*(t - d1) if t > d1 else 0
    C2_fn = lambda t : c2 * t # cumulative cost fns
    
    lookahead = lambda l1, l2: policy.Lookahead(d1 - np.log(mu1*c1/mu2/c2) / (mu1 - l1))
    accprios = [policy.AccPrio(b1, 1) for b1 in np.linspace(1, 44, 12)]
    policies = { "FCFS": [policy.FCFS], "AccPrio*": accprios,
                 r'gen-$c\mu$' : [policy.Lookahead(d1)], "Whittle": [lookahead],
                 'PPrio': [policy.PPrio12, policy.PPrio21]}
    costs_by_policy = {name: compute_best_costs(exp_name, mu1, mu2, C1_fn, C2_fn,
                        p1, policy_fam) for name, policy_fam in policies.items()}
    gen_plot(exp_name, costs_by_policy, p1)

def run_2deadline_exp(mu1, mu2, c1, c2, d1, d2, p1=0.25):
    # instantaneous c1(t) = c1 * is(t > d1), c2(t) = c2 * is(t>d2)
    exp_name = '2deadline_exp'
    c1_fn, C1_fn = lambda t : c1 if t > d1 else 0, lambda t : c1*(t - d1) if t > d1 else 0
    c2_fn, C2_fn = lambda t : c2 if t > d2 else 0, lambda t : c2*(t - d2) if t > d2 else 0

    age_values = np.linspace(0, max(d1, d2)*1.1, 20)
    gen_cmu = policy.generalized_cmu([mu1, mu2], [c1_fn, c2_fn], age_values)
    lookaheads = [policy.Lookahead(a) for a in np.linspace(0, d1, 12)]
    whittle = lambda l1, l2: policy.Whittle([l1, l2], [mu1, mu2], [c1_fn, c2_fn], age_values)
    fcfs = lambda l1, l2 : policy.FCFS
    accprios = [policy.AccPrio(b1, 1) for b1 in np.linspace(1, 44, 12)]
    policies = {'FCFS': [fcfs],  "AccPrio*":accprios,
                'PPrio':[policy.PPrio12, policy.PPrio21], 'Lookahead*' : lookaheads,
                r'gen-$c\mu$': [gen_cmu], 'Whittle': [whittle]}
    
    costs_by_policy = {name: compute_best_costs(exp_name, mu1, mu2, C1_fn, C2_fn,
                            p1, policy_fam) for name, policy_fam in policies.items()}
    gen_plot(exp_name, costs_by_policy, p1)    

def run_polynomial_cost_exp(mu1, mu2, c1, c2, p1=0.25):
    # instantaneous c1(t) = c1 t, c2(t) = c2 t^2
    exp_name = 'polynomial_cost_exp'
    C1_fn, C2_fn = lambda t : c1 * t**2 / 2, lambda t : c2 * t**3 / 3

    def whittle(l1, l2):
        cost_rate1, cost_rate2 = (0, c1, 0), (c2, 0, 0)
        return policy.QuadraticWhittle([l1, l2], [mu1, mu2], [cost_rate1, cost_rate2])
    def gen_cmu(l1, l2):
        gencmu = policy.QuadraticAccPrio([0, c2], [c1, 0], [0, 0])
        gencmu.policy_name = r'gen-$c\mu$'
        return gencmu
    accprios = [policy.AccPrio(b1, 1) for b1 in np.linspace(1, 44, 12)]
    lookaheads = [policy.Lookahead(a) for a in np.linspace(0, 2 * c1/c2, 12)]    
    policies = {'Whittle': [whittle], r'gen-$c\mu$': [gen_cmu], 'FCFS':[policy.FCFS],
                'AccPrio*': accprios, 'PPrio': [policy.PPrio12, policy.PPrio21],
                'Lookahead*': lookaheads}

    costs_by_policy = {name: compute_best_costs(exp_name, mu1, mu2, C1_fn, C2_fn,
                            p1, policy_fam) for name, policy_fam in policies.items()}
    gen_plot(exp_name, costs_by_policy, p1)    

if __name__ == "__main__":
    # given S1 ~ exp(mu1), S2 ~ exp(mu2), cost rate "constants" c1, c2
    # deadline/cost parameters d1, d2, gen plots of load -> cost for policies
    mu1, mu2, c1, c2, d1, d2 = 3, 1, 10, 1, 8, 4
    for p1 in [0.5]:
        #run_1deadline_exp(mu1, mu2, c1, c2, d1, p1)
        #run_2deadline_exp(mu1, mu2, c1, c2, d1, d2, p1)
        run_polynomial_cost_exp(mu1, mu2, 5, c2, p1)
        pass
    #gen_plot('polynomial_cost_exp', None, p1 = 0.5)
    #gen_plot('1deadline_exp', None, p1=0.5)
    #en_plot('2deadline_exp', None, p1=0.5)
    plt.show()
