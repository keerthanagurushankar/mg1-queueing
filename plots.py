import math, random, numpy as np
import matplotlib.pyplot as plt
import lib, simulations, policies as policy
import json

def save_sample_path(l1, l2, S1, S2, b1, b2, dname):
    PAccPrio = policy.AccPrio(b1, b2, is_preemptive = True)
    simulated_MG1 = simulations.MG1([simulations.JobClass(1, l1, S1), 
                                     simulations.JobClass(2, l2, S2)], PAccPrio)
    simulated_MG1.run()
    simulated_MG1.save_metrics(dname)    

def run_PAccPrio(l1, l2, S1, S2, b1, b2, dname=None):
    PAccPrio = policy.AccPrio(b1, b2, is_preemptive = True)
    simulated_MG1 = simulations.MG1([simulations.JobClass(1, l1, S1, dname), 
                                     simulations.JobClass(2, l2, S2, dname)], PAccPrio)
    simulated_MG1.run()
    response_times = [job['response_time'] for job in simulated_MG1.metrics]
    response_times_sq = [t**2 for t in response_times]
        
    return np.mean(response_times_sq)
    
def plot_b1_vs_ETsq(exp_name, l1, l2, mu1, mu2, b1_max=15, from_file=False):
    if not from_file:
        S1, S2 = lib.exp(mu1), lib.exp(mu2)
        save_sample_path(l1, l2, S1, S2, 1, 1, exp_name)
        
        b1s, ETsqs = [], []
        for i in range(b1_max):
            b1, b2 = np.exp(i/2), 1
            ETsq = run_PAccPrio(l, l, S1, S2, b1, b2, exp_name)   
            print(f"Ran computations for {i}, {b1} -> {ETsq}")             
            
            b1s.append(b1)
            ETsqs.append(ETsq)
            
        with open(f'{exp_name}/b1-vs-ETsq-values.json', 'a') as f:
            json.dump(list(zip(b1s, ETsqs)), f)
    else:
        b1_vs_ETsqs = json.load(open(f'{exp_name}/b1-vs-ETsq-values.json', 'r'))
        b1s, ETsqs = list(zip(*b1_vs_ETsqs))

    plt.figure()
    plt.plot(b1s, ETsqs)
    plt.xscale('log')
    plt.xlabel('$b_1$')
    plt.ylabel('$E[T^2]$')
    plt.title(f'Acc. Priority: λ1 = λ2 = {l1}, μ1={mu1}, μ2={mu2}, b2=1')
    plt.savefig(f'{exp_name}/b1-vs-ETsq.png')
    
if __name__ == "__main__":
    l, mu2 = 1, 2
    for mu1 in [3, 4, 5, 6, 8, 10]:
        S1, S2 = lib.exp(mu1), lib.exp(mu2) 
        plot_b1_vs_ETsq(f'MM1-{l}-{mu1}-{mu2}', l, l, mu1, mu2, from_file=True)
    plt.show()
