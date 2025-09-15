import math, random, numpy as np
import matplotlib.pyplot as plt
import lib, policies as policy
import simulations.event_based as simulations
import json, os, logging
from typing import Callable, List, Tuple, Dict
plt.rcParams.update({'font.size': 16})

# HELPER FUNCTIONS

def save_sample_path(arrival_rates: List[float], job_sizes: List[Callable], dname: str):
    job_classes = [simulations.JobClass(i+1, l, S) for i, (l, S) in 
                   enumerate(zip(arrival_rates, job_sizes))]
    simulated_MG1 = simulations.MG1(job_classes, policy.FCFS)
    simulated_MG1.run()
    simulated_MG1.save_metrics(dname)

def run_simulation(arrival_rates: List[float], job_sizes: List[Callable], policy, dname=None, save_metrics = False) -> Tuple[List[float]]:
    # run sim: if dname given, use dname/arrival_sequence{1,2}.json to gen sample path
    # return lists of response time samples from sim, one per class
    job_classes = [simulations.JobClass(i+1, l, S, dname) for i, (l, S) in 
                   enumerate(zip(arrival_rates, job_sizes))]
    simulated_MG1 = simulations.MG1(job_classes, policy)
    simulated_MG1.run()
    
    Tks = tuple(
        [job['response_time'] for job in simulated_MG1.metrics if job['job_class'] == k]
        for k in range(1, len(job_classes)+1)
    )

    if save_metrics:
        simulated_MG1.save_metrics(dname+f'/{policy.policy_name}')
    
    return Tks

def load_metrics(dname: str, policies: List[str]) -> List[Dict]:
    all_metrics = []
    for policy in policies:
        metrics = json.load(open(f'{dname}/{policy}/metrics.json', 'r'))
        all_metrics.append(metrics)
    return all_metrics

def map_response_times_to_costs(Cks: List[Callable], Tks: List[List[float]]) -> List[List[float]]:
    CkTks : List[List[float]] = [list(map(Cks[i], Tks[i])) for i in range(len(Cks))]
    return CkTks

def get_costs(exp_names: List[str]):
    all_costs_by_policy = []
    for exp_name in exp_names:
        # load json from exp_name/runs/costs_by_policy.json
        costs_file = f'{exp_name}/runs/costs_by_policy.json'
        if os.path.exists(costs_file):
            with open(costs_file, 'r') as f:
                old_costs_by_policy = json.load(f)
            all_costs_by_policy += old_costs_by_policy
    return all_costs_by_policy

def compute_cost_ratios(costs_by_policy: List[Dict[str, List[float]]]):
    policies = ["FCFS", "Whittle", "Aalto", r'gen-$c\mu$', "PPrio"] #costs_by_policy[0].keys()
    baseline_policy = 'Whittle'
    num_runs = len(costs_by_policy)
    cost_ratios = {policy: np.array([costs_by_policy[run][policy] for run in range(num_runs)]) /
                           np.array([costs_by_policy[run][baseline_policy] for run in range(num_runs)])
                             for policy in policies}
    # costs_by_policy[run][policy] : rho -> ECost[run[rho]]
    return cost_ratios

## MAIN FUNCTIONS

def compute_best_cost_for_rho(
        exp_name: str, 
        mus: List[float], 
        Cs: List[Callable], 
        ps: List[float], 
        rho: float, 
        policy_fam: List[policy.Policy]) -> float:
    """
    Given
        mus: list of service rates
        Cs: list of cumulative cost functions
        ps: list of class proportions (arrivals)
        rho: load
        policy_fam: list of policies to choose from
    Return time avg cost obtained by best policy in policy_fam
    """ 

    l = rho / sum(ps/mus)
    arrival_rates = l * ps
    job_sizes = [lib.exp(mu) for mu in mus]

    sample_name = f'{exp_name}/MM1-{rho:.3f}'
    if not os.path.exists(sample_name):
        save_sample_path(arrival_rates, job_sizes, sample_name)
    
    
    # run simulation for a given arrival sequence and policy, compute time avg cost
    def compute_cost_for_policy(policy: policy.Policy, save_metrics = False) -> float:
        policy = policy(arrival_rates) if callable(policy) else policy
        Tks : List[List[float]] = run_simulation(arrival_rates, job_sizes, policy, sample_name)
        classwise_jobavg_costs : List[float] = list(map(np.mean, map_response_times_to_costs(Cs, Tks)))
        #[np.mean(list(map(Ci, Ti))) for (Ci, Ti) in zip(Cs, Tks)]
        time_avg_cost = sum(arrival_rates * classwise_jobavg_costs)
        logging.info(f"Ran simulation for load {round(rho, 3)}, "
                     f"{policy.policy_name} -> {time_avg_cost}")        
        return time_avg_cost
    
    def cost_for_idx(idx: int) -> float:
        return compute_cost_for_policy(policy_fam[idx])
    
    # assume cost is unimodal in policy_fam[left_idx: right_idx+1], binary search argmin     
    def search_best_costs(left_idx, right_idx, left_cost=None, right_cost=None) -> float:
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

def compute_best_costs(exp_name: str, rhos: List[float], mus: List[float],  
        cs: List[Callable], ps: List[float], policy_fam: List[policy.Policy]) -> List[float]:
    rhos, mus, ps = np.array(rhos), np.array(mus), np.array(ps)
    return [compute_best_cost_for_rho(exp_name, mus, cs, ps, rho, policy_fam) for rho in rhos]

def gen_plot(exp_name: str, rhos: List[float], costs_by_policy: Dict[str, List[float]]):
    # given costs_by_policy: dict[str->list[list[float]]] (or read from file)
    costs_fname = f'{exp_name}/costs.json'
    if costs_by_policy:
        json.dump(costs_by_policy, open(costs_fname, 'w'))
    else:
        costs_by_policy = json.load(open(costs_fname, 'r'))

    plt.figure()
    plot_style = {'FCFS':('-', 2, 'blue', 0.7),
                  r'gen-$c\mu$': ('--', 2, 'green', 0.7),
                  'PPrio':(':', 2, 'purple', 1),                
                  'Aalto':('-', 2, 'orange', 0.7),
                  'Whittle':('--', 2, 'red', 0.7),
                  }

    for policy in plot_style:
        if policy in costs_by_policy:
            costs = costs_by_policy[policy]
            ls, lw, color, alpha = plot_style[policy]

            if policy == "Whittle":
                policy = "Us"

            if policy == "PPrio":
                policy = "Prio"

            plt.plot(
                rhos, costs,
                label=policy,linestyle=ls, linewidth=lw, color=color, alpha=alpha)

    plt.xlabel('Load')
    plt.ylabel('Time-avg Total Holding Cost')
    plt.legend(loc = "upper left")
    plt.savefig(f'{exp_name}/rhos-vs-costs.png')
