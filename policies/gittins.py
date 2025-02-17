import numpy as np
from .age_based import AgeBasedPrio, Whittle
import lib
#import time_based_sims as simulations
#import simulations.event_based as simulations
import simulations.event_based_FCFS as simulations
import logging
import matplotlib.pyplot as plt

def ECost(arrival_rates, holding_cost_rates, sim_res):
    s = 0.0
    for l, c, Ti in zip(arrival_rates, holding_cost_rates, sim_res):
        s += l*np.mean([c(T) for T in Ti])
    return s

def MG1_ECost_tests(test_label, arrival_rates, service_rates, cum_holding_cost_rates, policy):
    job_classes = [simulations.JobClass(n+1, arrival_rates[n], lib.exp(service_rates[n]))
                   for n in range(len(arrival_rates))]
    simulated_MG1 = simulations.MG1(job_classes, policy)    
    simulated_MG1.run()
    sim_res = [[job['response_time'] for job in simulated_MG1.metrics
                if job['job_class'] == n+1] for n in range(len(arrival_rates))]
        
    logging.info(f"SIMULATED: {test_label} Expected Cost: {ECost(arrival_rates, cum_holding_cost_rates, sim_res)}")
    for n in range(len(arrival_rates)):
        T = sim_res[n]
        logging.info(f"SIMULATED: {test_label} Class {n+1} mean: {np.mean(T)}, var.: {np.var(T)}")

def interp_NaN(age_values, arr):
    mask = np.isnan(arr)
    arr_int = np.interp(age_values[mask], age_values[~mask], arr[~mask])
    arr[mask] = arr_int
    return arr

# Generates V_i(t) given mu, C, and Ti
def gittins(mu, c, C, Ti, t):
    filtered = list(filter(lambda x: x >= t, Ti))
    if len(filtered) == 0:
        # logging.info(f"WARNING: No data at t = {t}")
        return np.nan #Default value if no data
    exc = np.mean([C(T) for T in filtered])
    ex = np.mean([T for T in filtered])
    return mu * (exc-C(t)) / (ex-t)

# Generates V_i(t) given mu, c, and Ti
def inst_gittins(mu, c, C, Ti, t):
    filtered = list(filter(lambda x: x >= t, Ti))
    if len(filtered) == 0:
        # logging.info(f"WARNING: No data at t = {t}")
        return np.nan #Default value if no data
    return mu * np.mean([c(T) for T in filtered])

def gittinsV_val(service_rates, inst_holding_cost_rates, cum_holding_cost_rates, sim_res,
            age_values=np.arange(0, 15, 0.1), gttns_fn=gittins):
    # list[float] * list[float -> float] * list[list[float]] -> Policy
    V_values = []
    for mu, c, C, Ti in zip(service_rates, inst_holding_cost_rates, cum_holding_cost_rates, sim_res):
        Vi_values = np.array([gttns_fn(mu, c, C, Ti, t) for t in age_values])
        Vi_values = interp_NaN(age_values, Vi_values)
        V_values.append(Vi_values)
    
    V_values = np.array(V_values)
    return V_values

# Iteratively generates gittins policy by running simulations and updating policy
def iterativeGittins_V(arrival_rates, service_rates, inst_holding_cost_rates, cum_holding_cost_rates, maxItr=10, alpha=0.2,
            age_values=np.arange(0, 15, 0.1), initialPolicy=None, gttns_fn=gittins):
    # Initialize policy and V values, defaults to Whittle
    if initialPolicy is None: policy = Whittle(arrival_rates, service_rates, inst_holding_cost_rates, age_values)
    else: policy = initialPolicy
    
    V_values = []
    for k in range(len(arrival_rates)):
        Vi_values = np.array([policy.priority_fn(None, None, t, k+1) for t in age_values])
        V_values.append(Vi_values)
    V_values = np.array(V_values)
    V = lambda r, s, t, k : np.interp(t, age_values, V_values[k-1])
    policy = AgeBasedPrio(V, age_values)

    itr = 0
    job_classes = [simulations.JobClass(n+1, arrival_rates[n], lib.exp(service_rates[n]))
                   for n in range(len(arrival_rates))]
    
    # maxItr iterations of updating the policy with gittins
    while(itr < maxItr):
        # Simulating with current policy
        itr += 1
        simulated_MG1 = simulations.MG1(job_classes, policy)    
        simulated_MG1.run()
        sim_res = [[job['response_time'] for job in simulated_MG1.metrics
                    if job['job_class'] == n+1] for n in range(len(arrival_rates))]
        
        # Printing simulation results
        logging.info(f"____ Iteration: {itr} ____")
        logging.info(f"SIMULATED: Expected Cost: {ECost(arrival_rates, cum_holding_cost_rates, sim_res)}")
        for n in range(len(arrival_rates)):
            T = sim_res[n]
            logging.info(f"SIMULATED: Class {n+1} mean: {np.mean(T)}, var.: {np.var(T)}")

        # Updating the policy
        V_new = gittinsV_val(service_rates, inst_holding_cost_rates, cum_holding_cost_rates, sim_res, age_values, gttns_fn)
        V_values = (1-alpha)*V_values + alpha*V_new
        V = lambda r, s, t, k : np.interp(t, age_values, V_values[k-1])
        policy = AgeBasedPrio(V, age_values)

    return V

# Iteratively generates gittins policy by running simulations and updating policy
def iterativeGittins(arrival_rates, service_rates, inst_holding_cost_rates, cum_holding_cost_rates, maxItr=10, alpha=0.2,
            age_values=np.arange(0, 15, 0.1), initialPolicy=None, gttns_fn=gittins):
    V = iterativeGittins_V(arrival_rates, service_rates, inst_holding_cost_rates, cum_holding_cost_rates, maxItr, alpha,
            age_values, initialPolicy, gttns_fn)
    return AgeBasedPrio(V, age_values)

def plotGittinsV(fileName, arrival_rates, service_rates, inst_holding_cost_rates, cum_holding_cost_rates, maxItr=10, alpha=0.2,
            age_values=np.arange(0, 15, 0.1), initialPolicy=None, gttns_fn=gittins):
    V = iterativeGittins_V(arrival_rates, service_rates, inst_holding_cost_rates, cum_holding_cost_rates, maxItr, alpha,
            age_values, initialPolicy, gttns_fn)
    for k in range(len(arrival_rates)):
        plt.plot(age_values, V(None, None, age_values, k), label=f"Class {k+1}")
    plt.xlabel("T")
    plt.ylabel("r*")
    plt.title("r*(T)")
    plt.legend()
    plt.savefig(fileName)
    plt.show()

if __name__ == "__main__":
    # Plotting r*
    l1, l2, mu1, mu2 = 3/8, 3/8, 3, 1
    c1, C1 = lambda t : 3 if t > 10 else 0, lambda t : 0 if t < 10 else 3*(t-10)
    c2, C2 = lambda t : 1 if t > 5 else 0, lambda t : 0 if t < 5 else 1*(t-5)

    plotGittinsV('gittinsR.png', [l1, l2], [mu1, mu2], [c1, c2], [C1, C2], 10, initialPolicy=None)
