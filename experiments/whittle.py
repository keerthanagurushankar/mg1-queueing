import math, random, numpy as np
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import policies as policy
from experiments import utils
import matplotlib.pyplot as plt
import json

rhos = np.linspace(0.8, 1, 15)[:-1]
accprio_b1s = np.logspace(-np.log(5), np.log(5), 20, base=np.e) # np.linspace(1, 30, 12)

# EXPERIMENT FUNCTIONS

def run_1deadline_exp(mu1, mu2, c1, c2, d1, p1=0.5):
    # instantaneous c1(t) = c1 * is(t > d1), c2(t) = c2
    exp_name = '1deadline_exp4' #3
    C1_fn = lambda t : c1*(t - d1) if t > d1 else 0
    C2_fn = lambda t : c2 * t # cumulative cost fns

    lookahead = lambda ar: policy.Lookahead(d1 - np.log(mu1*c1/mu2/c2) / (mu1 - ar[0]))
    aalto = lambda ar: policy.Lookahead(d1 - np.log(mu1*c1/mu2/c2) / (mu1))
    accprios = [policy.AccPrio(b1, 1) for b1 in accprio_b1s]
    policies = { "FCFS": [policy.FCFS],
                 #"AccPrio*": accprios,
                 "Whittle": [lookahead],
                 "Aalto": [aalto],
                 r'gen-$c\mu$' : [policy.Lookahead(d1)],
                 'PPrio': [policy.PPrio12, policy.PPrio21]}
    costs_by_policy = {name: utils.compute_best_costs(exp_name, rhos, [mu1, mu2], 
        [C1_fn, C2_fn], [p1, 1-p1], policy_fam) for name, policy_fam in policies.items()}
    utils.gen_plot(exp_name, rhos, costs_by_policy)

def run_2deadline_exp(mu1, mu2, c1, c2, d1, d2, p1=0.5):
    # instantaneous c1(t) = c1 * is(t > d1), c2(t) = c2 * is(t>d2)
    exp_name = '2deadline_exp_balanced4'#2, 3
    c1_fn, C1_fn = lambda t : c1 if t > d1 else 0, lambda t : c1*(t - d1) if t > d1 else 0
    c2_fn, C2_fn = lambda t : c2 if t > d2 else 0, lambda t : c2*(t - d2) if t > d2 else 0

    age_values = np.linspace(0, max(d1, d2)*1.1, 20)
    gen_cmu = policy.generalized_cmu([mu1, mu2], [c1_fn, c2_fn], age_values)
    whittle = lambda ar: policy.Whittle(ar, [mu1, mu2], [c1_fn, c2_fn], age_values)
    aalto = lambda ar: policy.Aalto(ar, [mu1, mu2], [c1_fn, c2_fn], age_values)
    a_max = d1 - np.log(mu1*c1/mu2/c2)/mu1

    # (^if d2=l1=0, opt lookahead is latest we should prioritize class 1 in 2 deadline case)
    lookaheads = [policy.Lookahead(a) for a in np.linspace(0, a_max, 12)]
    accprios = [policy.AccPrio(b1, 1) for b1 in accprio_b1s]
    policies = {'FCFS': [policy.FCFS],
                #"AccPrio*": accprios,
                'PPrio':[policy.PPrio12, policy.PPrio21],
                #'Lookahead*' : lookaheads,
                r'gen-$c\mu$': [gen_cmu],
                'Whittle': [whittle],
                'Aalto': [aalto]
                }
    
    costs_by_policy = {name: utils.compute_best_costs(exp_name, rhos, [mu1, mu2], 
        [C1_fn, C2_fn], [p1, 1-p1], policy_fam) for name, policy_fam in policies.items()}
    utils.gen_plot(exp_name, rhos, costs_by_policy)

def run_linear_cost_exp(mu1, mu2, c1, c2, p1=0.5):
    # instantaneous c1(t) = c1 t, c2(t) = c2 t
    exp_name = 'linear_cost_exp3'
    c1_fn, C1_fn = lambda t : c1 * t, lambda t : c1 * t**2 / 2
    c2_fn, C2_fn = lambda t : c2 * t, lambda t : c2 * t**2 / 2

    def whittle(ar):
        return policy.LinearWhittle(ar, [mu1, mu2], [(c1, 0), (c2, 0)])
    def aalto(ar):
        return policy.LinearAalto(ar, [mu1, mu2], [(c1, 0), (c2, 0)])
    def gen_cmu(ar):
        gencmu = policy.AccPrio(mu1 * c1, mu2 * c2, is_preemptive=True)
        gencmu.policy_name = r'gen-$c\mu$'
        return gencmu
    accprios = [policy.AccPrio(b1, 1) for b1 in np.linspace(1, 44, 4)]
    policies = {'Whittle': [whittle],
                r'gen-$c\mu$': [gen_cmu],
                'FCFS':[policy.FCFS],
                #'AccPrio*': accprios,
                'PPrio': [policy.PPrio12, policy.PPrio21],
                'Aalto': [aalto]}

    costs_by_policy = {name: utils.compute_best_costs(exp_name, rhos, [mu1, mu2], 
        [C1_fn, C2_fn], [p1, 1-p1], policy_fam) for name, policy_fam in policies.items()}
    utils.gen_plot(exp_name, rhos, costs_by_policy)

def run_polynomial_cost_exp(mu1, mu2, c1, c2, d1, p1=0.5):
    # instantaneous c1(t) = c1 t + d1, c2(t) = c2 t^2
    print(f'mu1 {mu1}, mu2 {mu2}, c1 {c1}, c2 {c2}, d1 {d1}, p1 {p1}')
    exp_name = 'polynomial_cost_exp3' #3
    C1_fn, C2_fn = lambda t : c1 * t**2 / 2 + d1 * t, lambda t : c2 * t**3 / 3

    def whittle(ar):
        cost_rate1, cost_rate2 = (0, c1, d1), (c2, 0, 0)
        return policy.QuadraticWhittle(ar, [mu1, mu2], [cost_rate1, cost_rate2])

    def aalto(ar):
        cost_rate1, cost_rate2 = (0, c1, d1), (c2, 0, 0)
        return policy.QuadraticAalto(ar, [mu1, mu2], [cost_rate1, cost_rate2])

    def gen_cmu(ar):
        gencmu = policy.QuadraticAccPrio([0, mu2 * c2], [mu1 * c1, 0], [mu1 * d1, 0])
        gencmu.policy_name = r'gen-$c\mu$'
        return gencmu

    accprios = [policy.AccPrio(1, b1) for b1 in accprio_b1s]

    policies = {
               
               'Whittle': [whittle],
               'Aalto': [aalto],
               r'gen-$c\mu$': [gen_cmu],
               
               'PPrio': [policy.PPrio12, policy.PPrio21],
               'FCFS':[policy.FCFS],
               'AccPrio*': accprios,
                }
    costs_by_policy = {name: utils.compute_best_costs(exp_name, rhos, [mu1, mu2], 
        [C1_fn, C2_fn], [p1, 1-p1], policy_fam) for name, policy_fam in policies.items()}
    utils.gen_plot(exp_name, rhos, costs_by_policy)

def run_multiclass_poly_exp(service_rates, cost_rates, p1=0.5):
    # ci(t) = ai t^2 + bi t + ci where (ai, bi, ci) in cost_rates
    exp_name = 'multiclass_exp4'#+str(round(np.random.rand(),2))

    # remember experiment parameters
    params = {'rhos':list(rhos),
              'service_rates':list(service_rates),
              'cost_rates':[list(cs) for cs in cost_rates]}
    json.dump(params, open(exp_name+'/params.json', 'w'))

    cumulative_cost_rates = [lambda t, c=cost_rates[i]:
                             c[0] * t**3/3 + c[1] * t**2/2 + c[2] * t
                             for i in range(len(cost_rates))]

    def whittle(arrival_rates):
        return policy.QuadraticWhittle(arrival_rates, service_rates, cost_rates)
    def aalto(arrival_rates):
        return policy.QuadraticAalto(arrival_rates, service_rates, cost_rates)
    gen_cmu = policy.QuadraticGenCMU(service_rates, cost_rates)

    acc_prios = [policy.LinearAccPrio([0, 0, 0], [b1, b2, b3])
                 for b1 in np.logspace(0, np.log(5), 4, base=np.e)
                 for b2 in np.logspace(0, np.log(5), 4, base=np.e)
                 for b3 in np.logspace(0, np.log(5), 4, base=np.e)]

    policies = {
                #'PPrio': policy.StrictPriorities(len(service_rates)),            
                #r'gen-$c\mu$': [gen_cmu],
                #'Whittle': [whittle],
                #'FCFS':[policy.FCFS],
                'AccPrio*': acc_prios,
                #'Aalto': [aalto]
                }

    k = len(service_rates)
    costs_by_policy = {name:  utils.compute_best_costs(exp_name, rhos, 
                        service_rates, cumulative_cost_rates, [1/k]*k, policy) 
                        for name, policy in policies.items()}
    utils.gen_plot(exp_name, rhos, costs_by_policy)


def run_multiclass_exp(exp_name, service_rates, cost_fns, cumulative_cost_fns, p1=0.5):
    #exp_name = 'multiclass_exp4'#+str(round(np.random.rand(),2))

    age_values = np.linspace(0, 20, 20)
    gen_cmu = policy.generalized_cmu(service_rates, cost_fns, age_values)
    whittle = lambda arrival_rates: policy.Whittle(
        arrival_rates, service_rates, cost_fns, age_values)
    aalto = lambda arrival_rates: policy.Aalto(
        arrival_rates, service_rates, cost_fns, age_values)

    policies = {#'PPrio': policy.PPrio12,
                'PPrio': policy.StrictPriorities(len(service_rates)),        
                r'gen-$c\mu$': gen_cmu,
                'Whittle': whittle,
                'FCFS':policy.FCFS,
                #'AccPrio*': accprios,
                'Aalto': aalto
                }
    
    k = len(service_rates)
    costs_by_policy = {name:  utils.compute_best_costs(exp_name, rhos, 
                        service_rates, cumulative_cost_fns, [1/k]*k, policy) 
                        for name, policy in policies.items()}
    utils.gen_plot(exp_name, rhos, costs_by_policy)

if __name__ == "__main__":
    # given S1 ~ exp(mu1), S2 ~ exp(mu2), cost rate "constants" c1, c2
    # deadline/cost parameters d1, d2, gen plots of load -> cost for policies
    mu1, mu2, c1, c2, d1, d2 = 3, 1, 5, 1, 10, 5

    #run_1deadline_exp(mu1, mu2, 10**3, c2, d1, 0.5)
    #run_2deadline_exp(mu1, mu2, 10**6, c2, d1, d2, 0.5)
    #run_1deadline_exp(3, 1, 10, 1, 5, 0.9)
    #run_2deadline_exp(3, 1, 10**6, 1, 10, 5, 0.5)
    #run_linear_cost_exp(3, 1, 1, 10, 0.9)
    
    #run_polynomial_cost_exp(3, 1, 1, 5, 0, 0.75)
    run_polynomial_cost_exp(3, 1, 1, 1, 30, 0.75)
    #run_1deadline_exp(3, 1, 10, 1, 2, 0.9)
    #run_2deadline_exp(3, 1, 10, 1, 10, 5, 0.5)

    #utils.gen_plot('polynomial_cost_exp3', None,  0.75)
    #plot.gen_plot2(['1deadline_exp3', '1deadline_exp4'], None, p1=0.9)
    #plot.gen_plot('1deadline_exp', None, p1=0.9)
    #utils.gen_plot('2deadline_exp', rhos, None)
    #plot.gen_plot('linear_cost_exp3', None, p1 = 0.1)
    #gen_plot('multiclass_exp4', None, p1=0.5)

    plt.show()
