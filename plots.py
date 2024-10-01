import math, random, numpy as np
import matplotlib.pyplot as plt
import lib, simulations, policies as policy
import json

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

def plot_best_b1s(l, mu2):
    for mu1 in [3.5, 4, 5]:
        assert l/mu1 + l/mu2 < 1, "Load must be less than 1"
        S1, S2 = lib.hyperexponential(mu1, Csq=10), lib.hyperexponential(mu2, Csq=10) 
        plot_b1_vs_ETsq(f'sample_paths/MH101-{l}-{mu1}-{mu2}', l, l, mu1, mu2,
                        from_file=False)   


def run_Lookahead(l1, l2, S1, S2, alpha, dname=None):
    simulated_MG1 = simulations.MG1([simulations.JobClass(1, l1, S1, dname), 
                                     simulations.JobClass(2, l2, S2, dname)],
                                     policy.Lookahead(alpha))
    simulated_MG1.run()
    response_times = [job['response_time'] for job in simulated_MG1.metrics]
    response_times_tail = [1 if t > 10 else 0 for t in response_times]
    return np.mean(response_times_tail)

def plot_alpha_vs_tail(dname, l1, l2, mu1, mu2, d=15, from_file=False):
    # for given (l1, l2, S1, S2) whose arrival seq is stored in dname/arrival_sequence{i}.json
    # plot alpha -> Pr[T>d] under policy P-Acc-Prio (b1, b2=1) and save figure
    if not from_file:
        S1, S2 = lib.exp(mu1), lib.exp(mu2)
        #save_sample_path(l1, l2, S1, S2, 1, 1, dname)
        alphas, tails = [], []
        
        for alpha in range(d+2):
            tail = run_Lookahead(l1, l2, S1, S2, alpha, dname)
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

def plot_best_alphas(l, mu2):
    for mu1 in [4]:
        plot_alpha_vs_tail(f'sample_paths/MM1-{l}-{mu1}-{mu2}', l, l, mu1, mu2)
        
if __name__ == "__main__":
    # plot_best_b1s(1, 1.5)
    plot_best_alphas(1, 1.5)
    plt.show()
