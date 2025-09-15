from . import utils
import numpy as np
import matplotlib.pyplot as plt

rhos = np.linspace(0.8, 1, 4)[:-1]

def run_gittins_exp(mu1, mu2, c1_fn, c2_fn, C1_fn, C2_fn, maxItr=10, alpha=0.2,
                    age_values=np.arange(0, 15, 0.1), p1=0.25):
    # instantaneous c1(t) = c1 * is(t > d1), c2(t) = c2 * is(t>d2)
    exp_name = 'gittins_exp'

    gen_cmu = policy.generalized_cmu([mu1, mu2], [c1_fn, c2_fn], age_values)
    whittle = lambda l1, l2: policy.Whittle([l1, l2], [mu1, mu2], [c1_fn, c2_fn], age_values)
    gittins = lambda l1, l2: policy.iterativeGittins([l1, l2], [mu1, mu2], [c1_fn, c2_fn], [C1_fn, C2_fn],
                                                     maxItr=maxItr, alpha=alpha, age_values=age_values)
    policies = {'FCFS': [policy.FCFS], # "AccPrio*":accprios,
                'PPrio':[policy.PPrio12, policy.PPrio21], #'Lookahead*' : lookaheads,
                r'gen-$c\mu$': [gen_cmu], 'Whittle': [whittle],
                'Gittins': [gittins]}
    
    costs_by_policy = {name: utils.compute_best_costs(exp_name, rhos, [mu1, mu2], 
        [C1_fn, C2_fn], [p1, 1-p1], policy_fam) for name, policy_fam in policies.items()}
    utils.gen_plot(exp_name, rhos, costs_by_policy)

def run_normalized_exp(mu1, mu2, c1_fn, c2_fn, C1_fn, C2_fn, maxItr=10, alpha=0.2,
                    age_values=np.arange(0, 15, 0.1), p1=0.25):
    # instantaneous c1(t) = c1 * is(t > d1), c2(t) = c2 * is(t>d2)
    exp_name = 'normalized_exp'

    whittle = lambda l1, l2: policy.Whittle([l1, l2], [mu1, mu2], [c1_fn, c2_fn], age_values)
    gittins = lambda l1, l2: policy.iterativeGittins([l1, l2], [mu1, mu2], [c1_fn, c2_fn], [C1_fn, C2_fn],
                                                     maxItr=maxItr, alpha=alpha, age_values=age_values)
    policies = {'FCFS': [policy.FCFS],
                'Whittle': [whittle],
                'Gittins': [gittins]
                }
    
    costs_by_policy = {name: utils.compute_best_costs(exp_name, rhos, [mu1, mu2], 
        [C1_fn, C2_fn], [p1, 1-p1], policy_fam) for name, policy_fam in policies.items()}
    
    normalized_costs = {name: [a/b for a,b in zip(costs_by_policy[name], costs_by_policy['FCFS'])]
                        for name, policy_fam in policies.items()}
    utils.gen_plot(exp_name, rhos, costs_by_policy)
    
if __name__ == "__main__":
    # given S1 ~ exp(mu1), S2 ~ exp(mu2), cost rate "constants" c1, c2
    # deadline/cost parameters d1, d2, gen plots of load -> cost for policies
    # mu1, mu2, c1, c2, d1, d2 = 3, 1, 5, 1, 10, 5
    # for p1 in [0.5]:
    #     #run_1deadline_exp(mu1, mu2, 10**6, c2, d1, p1)
    #     #run_2deadline_exp(mu1, mu2, 10**6, c2, d1, d2, p1)
    #     #run_polynomial_cost_exp(mu1, mu2, 1, c2, 30, p1)
    #     pass
    # #gen_plot('polynomial_cost_exp', None, p1 = 0.5)
    # gen_plot('1deadline_exp', None, p1=0.5)
    # gen_plot('2deadline_exp', None, p1=0.5)
    # plt.show()

    # # 2 deadline experiment including Gittins
    # c1_fn, C1_fn = lambda t : c1 if t > d1 else 0, lambda t : c1*(t - d1) if t > d1 else 0
    # c2_fn, C2_fn = lambda t : c2 if t > d2 else 0, lambda t : c2*(t - d2) if t > d2 else 0
    # age_values = np.linspace(0, max(d1, d2)*1.1, 20)
    # run_gittins_exp(mu1, mu2, c1_fn, c2_fn, C1_fn, C2_fn, maxItr=10, alpha=0.2, age_values=age_values, p1=0.5)
    # gen_plot('gittins_exp', None, p1=0.5)
    # plt.show()

    # Same experiment including Gittins
    mu1, mu2, c1, c2, d1, d2 = 2, 2, 3, 3, 7, 7
    c1_fn, C1_fn = lambda t : c1 if t > d1 else 0, lambda t : c1*(t - d1) if t > d1 else 0
    c2_fn, C2_fn = lambda t : c2 if t > d2 else 0, lambda t : c2*(t - d2) if t > d2 else 0
    age_values = np.linspace(0, max(d1, d2)*1.1, 20)

    # run_normalized_exp(mu1, mu2, c1_fn, c2_fn, C1_fn, C2_fn, maxItr=10, alpha=0.2, age_values=age_values, p1=0.25)
    run_normalized_exp(mu1, mu2, c1_fn, c2_fn, C1_fn, C2_fn, maxItr=100, alpha=0.07, age_values=age_values, p1=0.25)
    utils.gen_plot('normalized_exp', None, p1=0.25)
    plt.show()

    # # Plotting r*
    # l1, l2, mu1, mu2 = 3/8, 3/8, 3, 1
    # c1, C1 = lambda t : 3 if t > 10 else 0, lambda t : 0 if t < 10 else 3*(t-10)
    # c2, C2 = lambda t : 1 if t > 5 else 0, lambda t : 0 if t < 5 else 1*(t-5)
    # policy.plotGittinsV('gittinsR.png', [l1, l2], [mu1, mu2], [c1, c2], [C1, C2], maxItr=10, initialPolicy=None)

    # #Nomalized plot w/ high degree monomial
    # mu1, mu2, c1, c2= 2, 2, 3, 3
    # def mono(x): return lambda t : (x+1)*(t**x), lambda t : t**(x+1)
    # c1_fn, C1_fn = mono(10)
    # c2_fn, C2_fn = mono(10)

    # run_normalized_exp(mu1, mu2, c1_fn, c2_fn, C1_fn, C2_fn, maxItr=100, alpha=0.07, p1=0.25)
    # gen_plot('normalized_exp', None, p1=0.25)
    # plt.show()
