import math, random, numpy as np
import matplotlib.pyplot as plt
import lib, simulations, policies as policy
import json, os
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

def lambdas_from_rho(rho, mu1, mu2, p1=0.5):
    # given l1 = p1 l, rho = p1 * l / mu1 + (1-p1) * l /mu2
    l = rho / (p1 / mu1 + (1-p1) / mu2)
    l1, l2 = round(p1 * l, 3), round((1-p1)* l, 3)
    return l1, l2

def compute_costs_for_exp(exp_name, mu1, mu2, c1, c2, p1, policies):
    # given c1, c2: cumulative cost fns, p1 : l1 = p1 * l,
    # return costs : list[list[float]] : policy -> rho -> cost
    S1, S2 = lib.exp(mu1), lib.exp(mu2)
    ECosts = [] # list[list[float]] : rho -> pi -> cost

    for rho in rhos:
        l1, l2 = lambdas_from_rho(rho, mu1, mu2, p1)

        sample_name = f'{exp_name}/MM1-{l1}-{l2}-{mu1}-{mu2}'
        if not os.path.exists(sample_name):
            save_sample_path(l1, l2, S1, S2, sample_name)

        ECosts_for_rho = []
        for policy in policies:
            curr_policy = policy(l1, l2) if callable(policy) else policy
            T1, T2 = run_2class_simulation(l1, l2, S1, S2, curr_policy, sample_name)
            ECost =  np.mean(list(map(c1, T1)) + list(map(c2, T2)))
            print(f"Ran computations for {rho}, {ECost} <- {curr_policy.policy_name}")
            ECosts_for_rho.append(ECost)

        ECosts.append(ECosts_for_rho)

    ECosts = list(np.array(ECosts).T) # pi -> rho -> cost
    return ECosts

def compute_best_costs(exp_name, mu1, mu2, c1, c2, p1, policy_fam):
    costs = compute_costs_for_exp(exp_name, mu1, mu2, c1, c2, p1, policy_fam)
    best_costs = list(np.min(np.array(costs), axis=0))
    # costs : pi -> rho -> cost, costs*[rho] = min_pi xs[pi, rho]
    return best_costs

def gen_plot(exp_name, costs_by_policy, p1=0.25):
    # costs_by_policy: dict[str->list[float]]
    costs_fname = f'{exp_name}/costs{p1}.json'
    if costs_by_policy:
        json.dump(costs_by_policy, open(costs_fname, 'a'))
    else:
        costs_by_policy = json.load(open(costs_fname, 'r'))
    
    plt.figure()
    for policy, cost in costs_by_policy.items():
        plt.plot(rhos, cost, label=policy)
    
    plt.xlabel('Load')
    plt.ylabel('Cost')
    plt.legend()
    plt.savefig(f'{exp_name}/rhos-vs-costs-{p1}.png')

# EXPERIMENT FUNCTIONS

def run_1deadline_exp(mu1, mu2, c1, c2, d1, p1=0.25):
    exp_name = '1deadline_exp'
    cmu_ratio = mu1 * c1 / mu2 / c2 
    assert cmu_ratio > 1, "No priority for class 1"
    
    C1_fn = lambda t : c1*(t - d1) if t > d1 else 0
    C2_fn = lambda t : c2 * t # cumulative cost fns
    
    lookahead = lambda l1, l2: policy.Lookahead(d1 - np.log(mu1*c1/mu2/c2) / (mu1 - l1))
    accprios = [policy.AccPrio(b1, 1) for b1 in np.linspace(0.5 * cmu_ratio, 2 * cmu_ratio, 5)]
    policies = {"AccPrio*": accprios, "Whittle": [lookahead],  r'gen-$c\mu$' :
                [policy.Lookahead(d1)], "FCFS": [policy.FCFS]}
    costs_by_policy = {name: compute_best_costs(exp_name, mu1, mu2, C1_fn, C2_fn,
                        p1, policy_fam) for name, policy_fam in policies.items()}
    gen_plot(exp_name, costs_by_policy, p1)

def run_linear_cost_exp(mu1:float, mu2:float, c1:float, c2:float, p1=0.25):
    # instantaneous c(t) = c * t then cumulative C(t) = c * t **2 / 2
    exp_name = 'linear_cost_exp'
    cmu_ratio = mu1 * c1 / mu2 / c2
    assert cmu_ratio > 1, "No priority for class 1"
    
    c1_fn, C1_fn = lambda t : c1 * t, lambda t : c1 * t**2 / 2
    c2_fn, C2_fn = lambda t : c2 * t, lambda t : c2 * t**2 / 2
    
    def whittle(l1, l2):
        return policy.LinearWhittle([l1, l2], [mu1, mu2], [(c1, 0), (c2, 0)])
    def gen_cmu(l1, l2):
        gencmu = policy.AccPrio(mu1 * c1, mu2 * c2, is_preemptive=True)
        gencmu.policy_name = r'gen-$c\mu$'
        return gencmu
    accprios = [policy.AccPrio(b1, 1) for b1 in np.linspace(0.5 * cmu_ratio, 2 * cmu_ratio, 5)]
    policies = {'Whittle': [whittle], r'gen-$c\mu$': [gen_cmu], 'FCFS':[policy.FCFS],
                'AccPrio*': accprios}

    costs_by_policy = {name: compute_best_costs(exp_name, mu1, mu2, C1_fn, C2_fn,
                            p1, policy_fam) for name, policy_fam in policies.items()}
    gen_plot(exp_name, costs_by_policy, p1)
    
def run_quadratic_cost_exp(mu1, mu2, c1, c2, p1=0.25):
    exp_name = 'quadratic_cost_exp'
    C1_fn, C2_fn = lambda t : c1 * t**3 / 3, lambda t : c2 * t**3 / 3

    def whittle(l1, l2):
        cost_rate1, cost_rate2 = (c1, 0, 0), (c2, 0, 0) # c1_fn(t) = c1 * t**2        
        return policy.QuadraticWhittle([l1, l2], [mu1, mu2], [cost_rate1, cost_rate2])
    def gen_cmu(l1, l2):
        gencmu = policy.QuadraticAccPrio([c1, c2], [0, 0], [0, 0])
        gencmu.policy_name = r'gen-$c\mu$'
        return gencmu
    accprios = [policy.AccPrio(b1, 1) for b1 in np.linspace(15, 60, 5)]
    policies = {'Whittle': [whittle], r'gen-$c\mu$': [gen_cmu], 'FCFS':[policy.FCFS],
                'AccPrio*': accprios}

    costs_by_policy = {name: compute_best_costs(exp_name, mu1, mu2, C1_fn, C2_fn,
                            p1, policy_fam) for name, policy_fam in policies.items()}
    gen_plot(exp_name, costs_by_policy, p1)

def run_2deadline_exp(mu1, mu2, c1, c2, d1, d2, p1=0.25):
    exp_name = '2deadline_exp'
    c1_fn, C1_fn = lambda t : c1 if t > d1 else 0, lambda t : c1*(t - d1) if t > d1 else 0
    c2_fn, C2_fn = lambda t : c2 if t > d2 else 0, lambda t : c2*(t - d2) if t > d2 else 0

    gen_cmu = policy.generalized_cmu([mu1, mu2], [c1_fn, c2_fn], age_values=np.arange(0, 51, 1)/3)
    lookaheads = [policy.Lookahead(a) for a in np.linspace(20/3, 50/3, 6)[:-1]]
    whittle = lambda l1, l2: policy.Whittle([l1, l2], [mu1, mu2], [c1_fn, c2_fn], age_values=np.arange(0, 51, 1)/3)
    fcfs = lambda l1, l2 : policy.FCFS
    accprios = [policy.AccPrio(b1, 1) for b1 in np.linspace(15, 60, 5)]
    policies = {'FCFS': [fcfs], 'Lookahead*' : lookaheads, "AccPrio*" : accprios,
                r'gen-$c\mu$': [gen_cmu], 'Whittle': [whittle]}
    
    costs_by_policy = {name: compute_best_costs(exp_name, mu1, mu2, C1_fn, C2_fn,
                            p1, policy_fam) for name, policy_fam in policies.items()}
    gen_plot(exp_name, costs_by_policy, p1)

if __name__ == "__main__":
    mu1, mu2, c1, c2, d1, d2 = 3, 1, 10, 1, 50/3, 25/3
    # run_linear_cost_exp(mu1, mu2, c1, c2)
    # run_1deadline_exp(mu1, mu2, c1, c2, d1)
    # run_quadratic_cost_exp(mu1, mu2, c1, c2)
    run_2deadline_exp(mu1, mu2, c1, c2, d1, d2)
    #gen_plot('1deadline_exp', None, p1 = 0.25)
    plt.show()
