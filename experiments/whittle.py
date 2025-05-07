import math, random, numpy as np
import policies as policy
from .plots import gen_plot
import matplotlib.pyplot as plt

accprio_b1s = np.logspace(-np.log(5), np.log(5), 20, base=np.e) # np.linspace(1, 30, 12)

# EXPERIMENT FUNCTIONS

def run_1deadline_exp(mu1, mu2, c1, c2, d1, p1=0.5):
    # instantaneous c1(t) = c1 * is(t > d1), c2(t) = c2
    exp_name = '1deadline_exp4' #3
    C1_fn = lambda t : c1*(t - d1) if t > d1 else 0
    C2_fn = lambda t : c2 * t # cumulative cost fns

    lookahead = lambda l1, l2: policy.Lookahead(d1 - np.log(mu1*c1/mu2/c2) / (mu1 - l1))
    aalto = lambda l1, l2: policy.Lookahead(d1 - np.log(mu1*c1/mu2/c2) / (mu1))
    accprios = [policy.AccPrio(b1, 1) for b1 in accprio_b1s]
    policies = { "FCFS": [policy.FCFS],
                 #"AccPrio*": accprios,
                 "Whittle": [lookahead],
                 "Aalto": [aalto],
                 r'gen-$c\mu$' : [policy.Lookahead(d1)],
                 'PPrio': [policy.PPrio12, policy.PPrio21]}
    costs_by_policy = {name: plot.compute_best_costs(exp_name, mu1, mu2, C1_fn, C2_fn,
                        p1, policy_fam) for name, policy_fam in policies.items()}
    gen_plot(exp_name, costs_by_policy, p1)

def run_2deadline_exp(mu1, mu2, c1, c2, d1, d2, p1=0.5):
    # instantaneous c1(t) = c1 * is(t > d1), c2(t) = c2 * is(t>d2)
    exp_name = '2deadline_exp_balanced4'#2, 3
    c1_fn, C1_fn = lambda t : c1 if t > d1 else 0, lambda t : c1*(t - d1) if t > d1 else 0
    c2_fn, C2_fn = lambda t : c2 if t > d2 else 0, lambda t : c2*(t - d2) if t > d2 else 0

    age_values = np.linspace(0, max(d1, d2)*1.1, 20)
    gen_cmu = policy.generalized_cmu([mu1, mu2], [c1_fn, c2_fn], age_values)
    whittle = lambda l1, l2: policy.Whittle([l1, l2], [mu1, mu2], [c1_fn, c2_fn], age_values)
    aalto = lambda l1, l2: policy.Aalto([l1, l2], [mu1, mu2], [c1_fn, c2_fn], age_values)
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

    costs_by_policy = {name: plot.compute_best_costs(exp_name, mu1, mu2, C1_fn, C2_fn,
                            p1, policy_fam) for name, policy_fam in policies.items()}
    gen_plot(exp_name, costs_by_policy, p1)

def run_linear_cost_exp(mu1, mu2, c1, c2, p1=0.5):
    # instantaneous c1(t) = c1 t, c2(t) = c2 t
    exp_name = 'linear_cost_exp3'
    c1_fn, C1_fn = lambda t : c1 * t, lambda t : c1 * t**2 / 2
    c2_fn, C2_fn = lambda t : c2 * t, lambda t : c2 * t**2 / 2

    def whittle(l1, l2):
        return policy.LinearWhittle([l1, l2], [mu1, mu2], [(c1, 0), (c2, 0)])
    def aalto(l1, l2):
        return policy.LinearAalto([l1, l2], [mu1, mu2], [(c1, 0), (c2, 0)])
    def gen_cmu(l1, l2):
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

    costs_by_policy = {name: plot.compute_best_costs(exp_name, mu1, mu2, C1_fn, C2_fn,
                            p1, policy_fam) for name, policy_fam in policies.items()}
    gen_plot(exp_name, costs_by_policy, p1)

def run_polynomial_cost_exp(mu1, mu2, c1, c2, d1, p1=0.5):
    # instantaneous c1(t) = c1 t + d1, c2(t) = c2 t^2
    print(f'mu1 {mu1}, mu2 {mu2}, c1 {c1}, c2 {c2}, d1 {d1}, p1 {p1}')
    exp_name = 'polynomial_cost_exp3' #3
    C1_fn, C2_fn = lambda t : c1 * t**2 / 2 + d1 * t, lambda t : c2 * t**3 / 3

    def whittle(l1, l2):
        cost_rate1, cost_rate2 = (0, c1, d1), (c2, 0, 0)
        return policy.QuadraticWhittle([l1, l2], [mu1, mu2], [cost_rate1, cost_rate2])

    def aalto(l1, l2):
        cost_rate1, cost_rate2 = (0, c1, d1), (c2, 0, 0)
        return policy.QuadraticAalto([l1, l2], [mu1, mu2], [cost_rate1, cost_rate2])

    def gen_cmu(l1, l2):
        gencmu = policy.QuadraticAccPrio([0, mu2 * c2], [mu1 * c1, 0], [mu1 * d1, 0])
        gencmu.policy_name = r'gen-$c\mu$'
        return gencmu

    accprios = [policy.AccPrio(1, b1) for b1 in accprio_b1s]

    policies = {
               'AccPrio*': accprios,
               'Whittle': [whittle],
               # 'Aalto': [aalto],
               # r'gen-$c\mu$': [gen_cmu],
               # 
               # 'PPrio': [policy.PPrio12, policy.PPrio21],
               # 'FCFS':[policy.FCFS],
                }

    costs_by_policy = {name: plot.compute_best_costs(exp_name, mu1, mu2, C1_fn, C2_fn,
                            p1, policy_fam) for name, policy_fam in policies.items()}
    plot.gen_plot(exp_name, costs_by_policy, p1)



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
    #run_polynomial_cost_exp(3, 1, 1, 1, 30, 0.75)
    #run_1deadline_exp(3, 1, 10, 1, 2, 0.9)
    #run_2deadline_exp(3, 1, 10, 1, 10, 5, 0.5)

    #plot.gen_plot('polynomial_cost_exp3', None, p1 = 0.75)
    #plot.gen_plot2(['1deadline_exp3', '1deadline_exp4'], None, p1=0.9)
    #plot.gen_plot('1deadline_exp', None, p1=0.9)
    plot.gen_plot('2deadline_exp', None, p1=0.5)
    #plot.gen_plot('linear_cost_exp3', None, p1 = 0.1)
    #gen_plot('multiclass_exp4', None, p1=0.5)

    plt.show()
