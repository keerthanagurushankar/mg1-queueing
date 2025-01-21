import random, numpy as np
from policies import AgeBasedPrio, Whittle, FCFS
import lib, simulations
import logging
from tests import run_2Class_MG1_tests
import math

'''
#Generates the next gittins policy given simulation results
def nextGittins(service_rates, holding_cost_rates, sim_res,
            age_values=np.arange(0, 15, 0.1)):
    # list[float] * list[float -> float] * list[list[float]] -> Policy
    V_values = []
    for mu, c, Ti in zip(service_rates, holding_cost_rates, sim_res):
        Vi_values = np.array([gittins(mu, c, Ti, t) for t in age_values])
        V_values.append(Vi_values)
    
    V = lambda r, s, t, k : np.interp(t, age_values, V_values[k-1])
         #   for k in range(len(arrival_rates))]
    policy = AgeBasedPrio(V, age_values)
    policy.policy_name = "nextGittins"
    return policy
'''

alpha = 0.2

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

# Generates V_i(t) given mu, c, and Ti
def gittins(mu, c, Ti, t):
    # float * (float->float) * list[float] * float
    filtered = list(filter(lambda x: x >= t, Ti))
    if len(filtered) == 0:
        # logging.info(f"WARNING: No data at t = {t}")
        return np.nan #Default value if no data
    exc = np.mean([c(T) for T in filtered])
    ex = np.mean([T for T in filtered])
    return mu * (exc-c(t)) / (ex-t)

def gittinsV_val(service_rates, holding_cost_rates, sim_res,
            age_values=np.arange(0, 15, 0.1)):
    # list[float] * list[float -> float] * list[list[float]] -> Policy
    V_values = []
    for mu, c, Ti in zip(service_rates, holding_cost_rates, sim_res):
        Vi_values = np.array([gittins(mu, c, Ti, t) for t in age_values])
        Vi_values = interp_NaN(age_values, Vi_values)
        V_values.append(Vi_values)
    
    V_values = np.array(V_values)
    return V_values

# Iteratively generates gittins policy by running simulations and updating policy
def iterativeGittins(arrival_rates, service_rates, inst_holding_cost_rates, cum_holding_cost_rates, maxItr=10,
            age_values=np.arange(0, 15, 0.1)):
    # list[float] * list[float -> float] * maxItr -> Policy
    # Initial policy currently Whittle
    V_values = []
    for l, mu, c in zip(arrival_rates, service_rates, inst_holding_cost_rates):
        Ti = [random.expovariate(mu - l) for _ in range(10**4)]
        Vi_values = np.array([mu * np.mean([c(t + T) for T in Ti]) for t in age_values])
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
        V_new = gittinsV_val(service_rates, cum_holding_cost_rates, sim_res, age_values)
        V_values = (1-alpha)*V_values + alpha*V_new
        V = lambda r, s, t, k : np.interp(t, age_values, V_values[k-1])
        policy = AgeBasedPrio(V, age_values)

    return policy

if __name__ == "__main__":
    # # Fixed identical holding costs test of Gittins and FCFS
    # l1, mu1 = 3/8, 3
    # c1, C1 = lambda t : 5, lambda t : 5*t
    # l2, mu2, c2, C2 = l1, mu1, c1, C1
    
    # GittinsIdx = iterativeGittins([l1, l2], [mu1, mu2], [c1, c2], [C1, C2], 10)
    # run_2Class_MG1_tests("Gittins", l1, l2, lib.exp(mu1), lib.exp(mu2), GittinsIdx)
    # run_2Class_MG1_tests("FCFS", l1, l2, lib.exp(mu1), lib.exp(mu2), FCFS)

    # # One deadline identical classes test of Gittins and FCFS
    # l1, mu1 = 3/8, 3
    # c1, C1 = lambda t : 5 if t > 10 else 0, lambda t : 0 if t < 10 else 5*(t-10)
    # l2, mu2, c2, C2 = l1, mu1, c1, C1
    
    # GittinsIdx = iterativeGittins([l1, l2], [mu1, mu2], [c1, c2], [C1, C2], 10)
    # run_2Class_MG1_tests("Gittins", l1, l2, lib.exp(mu1), lib.exp(mu2), GittinsIdx)
    # run_2Class_MG1_tests("FCFS", l1, l2, lib.exp(mu1), lib.exp(mu2), FCFS)

    # # One deadline test of Gittins and Whittle
    # l1, l2, mu1, mu2 = 3/8, 3/8, 3, 1
    # c1, C1 = lambda t : 5 if t > 10 else 0, lambda t : 0 if t < 10 else 5*(t-10)
    # c2, C2 = lambda t : 1 if t > 5 else 0, lambda t : 0 if t < 5 else t-5

    # GittinsIdx = iterativeGittins([l1, l2], [mu1, mu2], [c1, c2], [C1, C2], 10)
    # run_2Class_MG1_tests("Gittins", l1, l2, lib.exp(mu1), lib.exp(mu2), GittinsIdx)
    # WhittleIdx = Whittle([l1, l2], [mu1, mu2], [c1, c2])
    # run_2Class_MG1_tests("Whittle", l1, l2, lib.exp(mu1), lib.exp(mu2), WhittleIdx)

    # Gittins vs. Whittle on weird functions
    l1, l2, mu1, mu2 = 3/8, 3/8, 3, 1
    dead1 = lambda t : 5 if t > 8 else 0, lambda t : 0 if t < 10 else 5*(t-8)
    dead2 = lambda t : 2 if t > 10 else (1 if t > 5 else 0), lambda t : 2*t-15 if t > 10 else (t-5 if t > 5 else 0)
    hw = lambda t : 5 if t > 3 and t < 7 else 0, lambda t : 20 if t > 7 else (5*(t-3) if t > 3 else 0)
    quad1 = lambda t : t*t-2*t+1, lambda t : t*t*t/3-t*t+t
    quad2 = lambda t : t*t-4*t+4, lambda t : t*t*t/3-2*t*t+4*t
    wave = lambda t : 1+math.sin(t), lambda t : 1+t-math.cos(t)
    bigWave = lambda t : 5*(1+math.sin(t)), lambda t : 5*(1+t-math.cos(t))
    osc = lambda t : math.exp(t) + math.sin(t), lambda t : math.exp(t) - math.cos(t)
    inv = lambda t : 1/(t+1), lambda t : math.log(t+1)
    invsq = lambda t : 1/((t+1)*(t+1)), lambda t : -1/(t+1)+1

    c1, C1 = hw
    c2, C2 = inv

    GittinsIdx = iterativeGittins([l1, l2], [mu1, mu2], [c1, c2], [C1, C2], 20)
    MG1_ECost_tests("Gittins", [l1, l2], [mu1, mu2], [C1, C2], GittinsIdx)
    WhittleIdx = Whittle([l1, l2], [mu1, mu2], [c1, c2])
    MG1_ECost_tests("Whittle", [l1, l2], [mu1, mu2], [C1, C2], WhittleIdx)