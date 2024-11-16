import math, random, numpy as np
import matplotlib.pyplot as plt
import lib, simulations, policies as policy
import json, os
from scipy import integrate

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

def get_rho_vs_cost(exp_name, mu1, mu2, c1, c2, policies, from_file=False):
    if not from_file:
        S1, S2 = lib.exp(mu1), lib.exp(mu2)
        rhos = np.linspace(0.5, 1, 10)[:-1]
        ECosts = [] # list[list[float]]
        #c1, c2, d1 = 10, 1, 50/3
        
        for rho in rhos:
            l1, l2 = lambdas_from_rho(rho, mu1, mu2, 0.25)
            
            sample_name = f'{exp_name}/MM1-{l1}-{l2}-{mu1}-{mu2}'
            if not os.path.exists(sample_name):
                save_sample_path(l1, l2, S1, S2, sample_name)
            
            ECosts_for_rho = []
            
            for policy in policies:
                curr_policy = policy(l1, l2)
                T1, T2 = run_2class_simulation(l1, l2, S1, S2, curr_policy, sample_name)
                ECost =  np.mean(list(map(c1, T1)) + list(map(c2, T2)))
                print(f"Ran computations for {rho}, {ECost} <- {curr_policy.policy_name}")
                ECosts_for_rho.append(ECost)
                
            ECosts.append(ECosts_for_rho)

        with open(f'{exp_name}/rhos-vs-costs-l1-1by4.json', 'w') as f:
            json.dump(list(zip(rhos, ECosts)), f)
    else:
        rhos_vs_ECosts = json.load(open(f'{exp_name}/rhos-vs-costs-l1-1by4.json', 'r'))
        rhos, ECosts = list(zip(*rhos_vs_ECosts))

    return rhos, ECosts

def gen_plot(exp_name:str, rhos:list[float], ECosts:list[list[float]], policies:list[str]):
    plt.figure()
    for policy, ECost in zip(policies, zip(*ECosts)):
        plt.plot(rhos, ECost, label=policy)
        
    plt.xlabel('Load')
    plt.ylabel('Cost')
    plt.legend()
    plt.savefig(f'{exp_name}/rhos-vs-costs-l1-1by4.png')

def run_linear_cost_exp(mu1:float, mu2:float, c1:float, c2:float):
    # instantaneous c(t) = c * t then cumulative C(t) = c * t **2 / 2
    exp_name = 'linear_cost_exp'
    c1_fn, C1_fn = lambda t : c1 * t, lambda t : c1 * t**2 / 2
    c2_fn, C2_fn = lambda t : c2 * t, lambda t : c2 * t**2 / 2
    
    def whittle(l1, l2):
        return policy.LinearWhittle([l1, l2], [mu1, mu2], [(c1, 0), (c2, 0)])
    def gen_cmu(l1, l2):
        gencmu = policy.AccPrio(mu1 * c1, mu2 * c2, is_preemptive=True)
        gencmu.policy_name = r'gen-$c\mu$'
        return gencmu
    fcfs = lambda l1, l2: policy.FCFS
    policies = {'Whittle': whittle, r'gen-$c\mu$': gen_cmu, 'FCFS':fcfs}
    
    rhos, ECosts = get_rho_vs_cost(exp_name, mu1, mu2, C1_fn, C2_fn, policies.values())
    gen_plot(exp_name, rhos, ECosts, policies.keys())

def run_1deadline_exp(mu1, mu2, c1, c2, d1):
    exp_name = '1deadline_exp2'
    assert mu1 * c1 > mu2 * c2, "No priority for class 1"
    #d1 = 500 # 1.1 * 5/mu1 # 1.1 * 5/mu1
    c1_fn, C1_fn = lambda t : c1 if t > d1 else 0, lambda t : c1*(t - d1) if t > d1 else 0
    c2_fn, C2_fn = lambda t : c2, lambda t : c2 * t
    gen_cmu = lambda l1, l2: policy.Lookahead(d1)
    lookahead = lambda l1, l2: policy.Lookahead(d1 - np.log(mu1*c1/mu2/c2) / (mu1 - l1))
    fcfs = lambda l1, l2 : policy.FCFS
    policies = {r'gen-$c\mu$': gen_cmu, 'Whittle': lookahead, 'FCFS': fcfs}
    
    rhos, ECosts = get_rho_vs_cost(exp_name, mu1, mu2, C1_fn, C2_fn, policies.values(), from_file=True)
    gen_plot(exp_name, rhos, ECosts, policies.keys())

def run_2deadline_exp(mu1, mu2, c1, c2, d1, d2):
    exp_name = '2deadline_exp'
    assert mu1 * c1 > mu2 * c2 and d1 > d2, "No priority for class 1"
    #d1 = 500 # 1.1 * 5/mu1 # 1.1 * 5/mu1
    c1_fn, C1_fn = lambda t : c1 if t > d1 else 0, lambda t : c1*(t - d1) if t > d1 else 0
    c2_fn, C2_fn = lambda t : c2 if t > d2 else 0, lambda t : c2*(t - d2) if t > d2 else 0
    gen_cmu = lambda l1, l2: policy.Lookahead(d1)
    lookahead = lambda l1, l2: policy.Lookahead(d1 - np.log(mu1*c1/mu2/c2) / (mu1 - l1))
    fcfs = lambda l1, l2 : policy.FCFS
    policies = {r'gen-$c\mu$': gen_cmu, 'Whittle': lookahead, 'FCFS': fcfs}
    
    rhos, ECosts = get_rho_vs_cost(exp_name, mu1, mu2, C1_fn, C2_fn, policies.values())
    gen_plot(exp_name, rhos, ECosts, policies.keys())

if __name__ == "__main__":
    #run_linear_cost_exp(1, 2, 3, 1)
    # mU = 3, muB = 1, cU = 10, cB = 1, d1 = 500/30
    run_1deadline_exp(mu1 = 3, mu2 = 1, c1 = 10, c2 = 1, d1 = 50/3)
    plt.show()
