import math, random, numpy as np
import matplotlib.pyplot as plt
import lib, simulations, policies as policy
import json
from scipy import integrate

def save_sample_path(l1, l2, S1, S2, b1, b2, dname):
    # run a PAccPrio simulation and save metrics to dname (we only use arrival seqs)
    PAccPrio = policy.AccPrio(b1, b2, is_preemptive = True)
    simulated_MG1 = simulations.MG1([simulations.JobClass(1, l1, S1), 
                                     simulations.JobClass(2, l2, S2)], PAccPrio)
    simulated_MG1.run()
    simulated_MG1.save_metrics(dname)    

def run_PAccPrio(l1, l2, S1, S2, b1, b2, dname=None):
    # run a PAccPrio sim (from dname if given) and return ETsq
    PAccPrio = policy.AccPrio(b1, b2, is_preemptive = True)
    simulated_MG1 = simulations.MG1([simulations.JobClass(1, l1, S1, dname), 
                                     simulations.JobClass(2, l2, S2, dname)], PAccPrio)
    simulated_MG1.run()
    response_times = [job['response_time'] for job in simulated_MG1.metrics]
    response_times_sq = [t**2 for t in response_times]
    return np.mean(response_times_sq)

def run_Lookahead(l1, l2, S1, S2, alpha, d, dname=None):
    # run a Lookahead sim and return tail
    simulated_MG1 = simulations.MG1([simulations.JobClass(1, l1, S1, dname), 
                                     simulations.JobClass(2, l2, S2, dname)],
                                     policy.Lookahead(alpha))
    simulated_MG1.run()
    response_times = [job['response_time'] for job in simulated_MG1.metrics]
    response_times_tail = [1 if t > d else 0 for t in response_times]
    return np.mean(response_times_tail)

def run_Whittle(l1, l2, mu1, mu2, c1, c2, dname=None):
    # run a Whittle sim and return time-avg holding cost
    simulated_MG1 = simulations.MG1([simulations.JobClass(1, l1, lib.exp(mu1), dname),
                                     simulations.JobClass(2, l2, lib.exp(mu2), dname)],
                                    policy.Whittle([l1, l2], [mu1, mu2], [c1, c2]))
    simulated_MG1.run()
    T1s = [job['response_time'] for job in simulated_MG1.metrics if job['job_class']==1]
    T2s = [job['response_time'] for job in simulated_MG1.metrics if job['job_class']==2]
    C1, C2 = lambda T1: integrate.quad(c1, 0, T1), lambda T2: integrate.quad(c2, 0, T2)
    ECT1, ECT2 = np.mean([C1(T1) for T1 in T1s]), np.mean([C2(T2) for T2 in T2s])
    return l1 * ECT1 + l2 * ECT2
    
def plot_b1_vs_ETsq(dname, l1, l2, mu1, mu2, b1_max=15, from_file=False):
    # for given (l1, l2, S1, S2) whose arrival seq is stored in dname/arrival_sequence{i}.json
    # plot b1 -> ETsq under policy P-Acc-Prio (b1, b2=1) and save figure
    if not from_file:
        S1, S2 = lib.exp(mu1), lib.exp(mu2)
        save_sample_path(l1, l2, S1, S2, 1, 1, dname)
        
        b1s, ETsqs = [], []
        for i in range(b1_max):
            b1, b2 = np.exp(i/2), 1
            ETsq = run_PAccPrio(l, l, S1, S2, b1, b2, dname)   
            print(f"Ran computations for {i}, {b1} -> {ETsq}")             
            
            b1s.append(b1)
            ETsqs.append(ETsq)
            
        with open(f'{dname}/b1-vs-ETsq-values.json', 'a') as f:
            json.dump(list(zip(b1s, ETsqs)), f)
    else:
        b1_vs_ETsqs = json.load(open(f'{dname}/b1-vs-ETsq-values.json', 'r'))
        b1s, ETsqs = list(zip(*b1_vs_ETsqs))

    plt.figure()
    plt.plot(b1s, ETsqs)
    plt.xscale('log')
    plt.xlabel(r'$b_1$')
    plt.ylabel(r'$E[T^2]$')
    plt.title(f'Acc. Priority: Csq = 10, λ1 = λ2 = {l1}, μ1={mu1}, μ2={mu2}, b2=1')
    plt.savefig(f'{dname}/b1-vs-ETsq.png')

def plot_alpha_vs_tail(dname, l1, l2, mu1, mu2, d=10, from_file=False):
    # for given (l1, l2, S1, S2) whose arrival seq is stored in dname/arrival_sequence{i}.json
    # and deadline d, plot alpha -> Pr[T>d] under policy Lookahead(alpha) and save figure
    if not from_file:
        S1, S2 = lib.exp(mu1), lib.exp(mu2)
        #save_sample_path(l1, l2, S1, S2, 1, 1, dname)
        alphas, tails = [], []
        
        for alpha in range(3 * d // 2):
            tail = run_Lookahead(l1, l2, S1, S2, alpha, d, dname)
            print(f"Ran computations for {alpha}, {alpha} -> {tail}")
            alphas.append(alpha)
            tails.append(tail)
            
        with open(f'{dname}/alpha-vs-tail-values.json', 'a') as f:
            json.dump(list(zip(alphas, tails)), f)
    else:
        alpha_vs_tails = json.load(open(f'{dname}/alpha-vs-tail-values.json', 'r'))
        alphas, tails = list(zip(*alpha_vs_tails))

    plt.figure()
    plt.plot(alphas, tails)
    plt.xlabel(r'$\alpha$')
    plt.ylabel(rf'$\Pr[T>{d}]$')
    plt.title(f'Lookahead, λ1 = λ2 = {l1}, μ1={mu1}, μ2={mu2}')
    plt.savefig(f'{dname}/alpha-vs-tails.png')

def plot_l1_vs_ECost(dname, l, mu, c, from_file=False):
    if not from_file:
        S1, S2 = lib.exp(mu), lib.exp(mu)
        l1s, ECosts = [], []

        for l1 in np.linspace(l/2, l, 5):
            ECost = run_Whittle(l1, l-l1, mu, mu, c, c, dname)
            print(f"Ran computations for {l1}, {l1} -> {ECost}")
            l1s.append(l1)
            ECosts.append(ECost)
            
        with open(f'{dname}/l1-vs-ECost-values.json', 'a') as f:
            json.dump(list(zip(l1s, ECosts)), f)

    else:
        l1s_vs_ECosts = json.load(open(f'{dname}/l1-vs-ECost-values.json', 'r'))
        l1s, ECosts = list(zip(*l1s_vs_ECosts))

    plt.figure()
    plt.plot(l1s, ECosts)
    plt.xlabel(r'$\lambda_1$')
    plt.ylabel(r'$\mathbb{E}[\operatorname{Cost}]$')
    plt.title(f'Whittle Suboptimality to FCFS')
    plt.savefig(f'{dname}/l1-vs-ECosts.png')

def plot_best_PAccPrio(l, mu2):
    for mu1 in [3.5, 4, 5]:
        assert l/mu1 + l/mu2 < 1, "Load must be less than 1"
        plot_b1_vs_ETsq(f'sample_paths/MH101-{l}-{mu1}-{mu2}', l, l, mu1, mu2,
                        from_file=False)   
    
def plot_best_Lookahead(l, mu2):
    for mu1 in [1]:
        assert l/mu1 + l/mu2 < 1, "Load must be less than 1"        
        plot_alpha_vs_tail(f'sample_paths/MM1-{l}-{mu1}-{mu2}', l, l, mu1, mu2)

def plot_worst_Whittle(l, mu, d):
    c = lambda t : 1 if t > d else 0
    assert l < 2 * mu
    plot_l1_vs_ECost(f'sample_paths/MM1-{l}-{mu}-{mu}', l, mu, c, from_file=True)
        
if __name__ == "__main__":
    # plot_best_PAccPrio(1, 1.5)
    # plot_best_Lookahead(0.5, 2)
    plot_worst_Whittle(1, 3, 1)
    plt.show()
