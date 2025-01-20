import math, random, numpy as np
import matplotlib.pyplot as plt
import lib, simulations, policies as policy, plots as plot
import json, os, logging
plt.rcParams.update({'font.size': 14})

# CONSTANTS

rhos = np.linspace(0.8, 1, 10)[:-1]
accprio_b1s = np.logspace(0, np.log(30), 12, base=np.e) #np.linspace(1, 30, 12)

# HELPER FUNCTIONS

def save_sample_path(arrival_rates, service_rates, dname):
    # run an FCFS simulation and save metrics to dname (we only use arrival seqs)
    job_classes = [simulations.JobClass(i+1, li, Si) for i, (li, Si)
                   in enumerate(zip(arrival_rates, service_rates))]
    simulated_MG1 = simulations.MG1(job_classes, policy.FCFS)
    simulated_MG1.run()
    simulated_MG1.save_metrics(dname)

def run_multiclass_simulation(arrival_rates, service_rates, policy, dname=None):
    # run sim: if dname given, use dname/arrival_sequence{1,2}.json to gen sample path
    # return 2 lists of response time samples from sim
    job_classes = [simulations.JobClass(i+1, li, Si, dname) for i, (li, Si)
                   in enumerate(zip(arrival_rates, service_rates))]
    simulated_MG1 = simulations.MG1(job_classes, policy)
    simulated_MG1.run()
    Tis = [[job['response_time'] for job in simulated_MG1.metrics if
            job['job_class'] == i] for i in range(1, len(arrival_rates)+1)]

    return Tis

def compute_costs(exp_name, service_rates, cost_rates, cumulative_cost_rates, rho, policies):
    li = rho / sum([1/mui for mui in service_rates])
    arrival_rates = [li] * len(service_rates)
    job_sizes = [lib.exp(mui) for mui in service_rates]

    sample_name = f'{exp_name}/MM1-{round(li, 3)}'
    if not os.path.exists(sample_name):
        save_sample_path(arrival_rates, job_sizes, sample_name)

    # run simulation for a given arrival sequence and list of policies
    ECosts = []
    for policy in policies:    
        policy = policy(arrival_rates) if callable(policy) else policy
        Tis = run_multiclass_simulation(arrival_rates, job_sizes, policy, sample_name)
        ECost = np.mean(sum([list(map(Ci, Ti)) for Ci, Ti
                    in zip(cumulative_cost_rates, Tis)], []))
        logging.info(f"Ran simulation for load {round(rho, 3)}, "
                     f"{policy.policy_name} -> {ECost}")
        ECosts.append(ECost)
        
    return min(ECosts)


def run_multiclass_poly_exp(service_rates, cost_rates, p1=0.5):
    # ci(t) = ai t^2 + bi t + ci where (ai, bi, ci) in cost_rates
    exp_name = 'multiclass_exp3'#+str(round(np.random.rand(),2))

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

    policies = {
                'PPrio': policy.StrictPriorities(len(service_rates)),            
                r'gen-$c\mu$': [gen_cmu],
                'Whittle': [whittle],
                'FCFS':[policy.FCFS],
                #'AccPrio*': accprios,
                'Aalto': [aalto]
                }

    costs_by_policy = {name:  [compute_costs(exp_name, service_rates, cost_rates,
                        cumulative_cost_rates, rho, policy) for rho in rhos]
                        for name, policy in policies.items()}
    plot.gen_plot(exp_name, costs_by_policy, p1)


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

    costs_by_policy = {name:  [compute_costs(exp_name, service_rates, cost_rates,
                        cumulative_cost_fns, rho, policy) for rho in rhos]
                        for name, policy in policies.items()}
    plot.gen_plot(exp_name, costs_by_policy, p1)


if __name__ == "__main__":
    # given S1 ~ exp(mu1), S2 ~ exp(mu2), cost rate "constants" c1, c2
    # deadline/cost parameters d1, d2, gen plots of load -> cost for policies
    mu1, mu2, c1, c2, d1, d2 = 3, 1, 5, 1, 10, 5

    k=3
    cost_rates = [np.random.rand(3) for _ in range(k)]
    #cost_rates = [(10, 0, 0), (0, 1, 100), (0, 1, 200)]
    #cost_rates = [(1, 0, 0), (0.5, 1, 2)]
    #cost_rates = [(0, 0, 0, 1), (100, 0, 0, 0)]
    #run_multiclass_poly_exp([1, 3, 3], cost_rates)
    #run_multiclass_poly_exp([1, 3, 3], cost_rates)
    run_multiclass_poly_exp([1]*k, cost_rates)

    #gen_plot('linear_cost_exp3', None, p1 = 0.5)
    plot.gen_plot('multiclass_exp2', None, p1=0.5)

    plt.show()
