import math, random, numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import policies as policy, lib
from .plots import gen_plot, gen_lookahead_amount_plot
import logging

def compute_lookahead_amount(mu1, mu2, c1_fn, c2, X_rate=0, max_age=20):
    # find t s.t. mu1 E[c1(t + X)] - mu2 c2 = 0
    X = [random.expovariate(X_rate) for _ in range(10**5)]
    overtake_cond = lambda t: mu1 * np.mean([c1_fn(t+Xi) for Xi in X]) - mu2 * c2
    if overtake_cond(0) <= 0 and overtake_cond(max_age) <= 0:
        logging.warn("Lookahead doesn't cross, check max_age")
        return float('inf')
    elif overtake_cond(0) >= 0 and overtake_cond(max_age) >= 0:        
        return 0
    else:
        lookahead_amount = opt.brentq(overtake_cond, 0, max_age)
        return lookahead_amount

def run_lookahead_exp(exp_name, mu1, mu2, c1_fn, C1_fn, c2, p1=0.5, max_age=20):
    C2_fn = lambda t : c2 * t
    
    aalto_amount = compute_lookahead_amount(mu1, mu2, c1_fn, c2, mu1, max_age)
    aalto = policy.Lookahead(aalto_amount)

    gen_cmu_amount = compute_lookahead_amount(mu1, mu2, c1_fn, c2, float('inf'), max_age)
    gen_cmu = policy.Lookahead(gen_cmu_amount)

    def lookahead(l1, l2):
        alpha_star = compute_lookahead_amount(mu1, mu2, c1_fn, c2, mu1-l1, max_age)
        return policy.Lookahead(alpha_star)

    policies = { "FCFS": [policy.FCFS],
                 #"AccPrio*": accprios,
                 "Whittle": [lookahead],
                 "Aalto": [aalto],
                 r'gen-$c\mu$' : [gen_cmu],
                 'PPrio': [policy.PPrio12, policy.PPrio21]}
    
    #costs_by_policy = {name: plot.compute_best_costs(exp_name, mu1, mu2, C1_fn, C2_fn,
                     #   p1, policy_fam) for name, policy_fam in policies.items()}
    #plot.gen_plot(exp_name, costs_by_policy, p1)

    gen_lookahead_amount_plot(exp_name, policies, mu1, mu2, p1)

    
if __name__ == "__main__":

    c1_fn = lambda t : 10 if t > 10 else 0
    C1_fn = lambda t : 10 * max(0, t-10)
    run_lookahead_exp('lookahead1', 3, 1, c1_fn, C1_fn, 1, 0.9)

    c1_fn = lambda t : 10**5 if t > 20 else 0
    C1_fn = lambda t : 10**5 * max(0, t-20)
    #run_lookahead_exp('lookahead2', 1, 1, c1_fn, C1_fn, 1, 0.5, max_age=70)

    c1_fn = lambda t : t**2
    C1_fn = lambda t : t**3/3
    #run_lookahead_exp('lookahead3b', 1, 3, c1_fn, C1_fn, 30, 0.75) # finished

    c1_fn = lambda t : t + math.sin(t)
    C1_fn = lambda t : t**2/2 + 1-math.cos(t)
    #run_lookahead_exp('lookahead4', 3, 1, c1_fn, C1_fn, 1, 0.75)

    c1_fn = lambda t : math.exp(t)
    C1_fn = lambda t : math.exp(t)-1
    #run_lookahead_exp('lookahead5', 3, 1, c1_fn, C1_fn, 1, 0.75)    

    #plot.gen_plot('lookahead1', None, p1=0.9)
    #plot.gen_plot('lookahead2', None, p1=0.5)
    #plot.gen_plot('lookahead3b', None, p1=0.75)
    #plot.gen_plot('lookahead4', None, p1=0.75)
    
    plt.show()

    
    
