import math, random, numpy as np
import matplotlib.pyplot as plt
import lib, simulations, policies as policy
import json
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
                                     simulations.JobClass(2, l2, S2, dname)], PAccPrio)
    simulated_MG1.run()
    T1 = [job['response_time'] for job in simulated_MG1.metrics if job.job_class == 1]
    T2 = [job['response_time'] for job in simulated_MG1.metrics if job.job_class == 2]
    return T1, T2


def get_rho_vs_cost(exp_name, mu1, mu2, c1, c2, policies, from_file=False):
    if not from_file:
        S1, S2 = lib.exp(mu1), lib.exp(mu2)
        rhos = np.linspace(0.5, 1, 5)[:-1]
        ECosts = []
        
        for rho in rhos:
            l = rho/(1/mu1 + 1/mu2)
            l1, l2 = round(l/2, 2), round(l/2, 2)
            sample_name = f'{dname}/MM1-{l1}-{l2}-{mu1}-{mu2}'
            ECosts_for_rho = []
            for policy in policies:
                T1, T2 = run_2class_simulation(l1, l2, S1, S2, policy, sample_name)
                ECost =  np.mean(list(map(c1, T1)) + list(map(c2, T2)))
                print(f"Ran computations for {rho, policy.policy_name} -> {ECost}")
                ECosts_for_rho.append(ECost)
                
            ECosts.append(ECosts_for_rho)

        with open(f'{exp_name}/rho-vs-ECosts.json', 'w') as f:
            json.dump(list(zip(rhos, ECosts)), f)
    else:
        rhos_vs_ECosts = json.load(open(f'{exp_name}/rhos-vs-ECosts.json', 'r'))
        rhos, ECosts = list(zip(*rhos_vs_ECosts))

    return rhos, ECosts

def gen_plot(exp_name, rhos, ECosts, policies):
    plt.figure()
    for policy, ECost in zip(policies, zip(*ECosts)):
        plt.plot(rhos, ECost, label=policy.policy_name)
        
    plt.xlabel('Load')
    plt.ylabel('Cost')
    plt.legend()
    plt.savefig(f'{exp_name}/rhos-vs-costs-l1=l2.png')

def linear_cost_exp(exp_name, mu1:float, mu2:float, c1:float, c2:float):
    # instantaneous c(t) = c * t then cumulative C(t) = c * t **2 / 2
    c1_fn, C1_fn = lambda t : c1 * t, lambda t : c1 * t**2 / 2
    c2_fn, C2_fn = lambda t : c2 * t, lambda t : c2 * t**2 / 2
    whittle = lambda l1, l2: policy.LinearWhittle([l1, l2], [mu1, mu2], [(c1, 0), (c2, 0)])
    gen_cmu = policy.generalized_cmu([mu1, mu2], [c1_fn, c2_fn])
    pass

def run_Whittle(l1, l2, mu1, mu2, c1, c2, dname=None):
    # run a Whittle sim and return time-avg holding cost
    simulated_MG1 = simulations.MG1([simulations.JobClass(1, l1, lib.exp(mu1), dname),
                                     simulations.JobClass(2, l2, lib.exp(mu2), dname)],
                                   # policy.Whittle([l1, l2], [mu1, mu2], [(c1, 0), (c2, 0)]))
                        policy.LinearWhittle([l1, l2], [mu1, mu2], [(c1, 0), (c2, 0)]))
    simulated_MG1.run()
    T1s = [job['response_time'] for job in simulated_MG1.metrics if job['job_class']==1]
    T2s = [job['response_time'] for job in simulated_MG1.metrics if job['job_class']==2]
    
    #C1, C2 = lambda T1: integrate.quad(c1, 0, T1), lambda T2: integrate.quad(c2, 0, T2)
    #ECT1, ECT2 = np.mean([C1(T1) for T1 in T1s]), np.mean([C2(T2) for T2 in T2s])
    T1s, T2s = np.array(T1s), np.array(T2s)
    ECT1, ECT2 = np.mean(c1 * T1s ** 2), np.mean(c2 * T2s ** 2)
    return l1 * ECT1 + l2 * ECT2
    

def plot_rho_vs_ECost(dname, mu1, mu2, c1, c2, from_file=False):
    if not from_file:
        rhos, ECosts = [], []
        for rho in np.linspace(0.5, 0.95, 5):
            l = rho / (1/mu1 + 1/mu2)
            l1 = 0.5 * l
            l2 = l - l1
            sample_path_dname = f'{dname}/MM1-{l1}-{l2}-{mu1}-{mu2}'
            save_sample_path(l1, l2, lib.exp(mu1), lib.exp(mu2), sample_path_dname)            
            ECost = run_Whittle(l1, l2, mu1, mu2, c1, c2, sample_path_dname)
            print(f"Ran computations for {l1}, {l1} -> {ECost}")
            rhos.append(rho)
            ECosts.append(ECost)
            
        with open(f'{dname}/rhos-vs-ECost-values.json', 'w') as f:
            json.dump(list(zip(rhos, ECosts)), f)

    else:
        rhos_vs_ECosts = json.load(open(f'{dname}/rhos-vs-ECost-values.json', 'r'))
        rhos, ECosts = list(zip(*rhos_vs_ECosts))

    plt.figure()
    plt.plot(rhos, ECosts)
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$\mathbb{E}[\operatorname{Cost}]$')
    plt.title(f'Whittle Suboptimality to FCFS')
    plt.savefig(f'{dname}/rhos-vs-ECosts.png')



def plot_LinearWhittle_cmp(mu1, mu2, c1, c2):
    #c = lambda t : 1 if t > d else 0
    #assert l < 2 * mu
    #plot_rho_vs_ECost(f'sample_paths/MM1-{l1}-{mu}-{mu}', l, mu, c, from_file=True)
    plot_rho_vs_ECost(f'Whittle_sample_paths', mu1, mu2, c1, c2)
    
        
if __name__ == "__main__":
    plot_LinearWhittle_cmp(1, 2, 3, 1)
    plt.show()
