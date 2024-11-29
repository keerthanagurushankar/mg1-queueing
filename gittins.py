import numpy as np
from policies import AgeBasedPrio, Whittle
import lib, simulations

#Generates V_i(t) given mu, c, and Ti
def gittins(mu, c, Ti, t):
    # float * (float->float) * list[float] * float
    exc = np.mean([c(T) for T in filter(lambda x: x >= t, Ti)])
    ex = np.mean([T for T in filter(lambda x: x >= t, Ti)])
    return mu * (exc-c(t)) / (ex-t)

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

#Iteratively generates gittins policy by running simulations and updating policy
def iterativeGittins(arrival_rates, service_rates, holding_cost_rates, maxItr,
            age_values=np.arange(0, 15, 0.1)):
    # list[float] * list[float -> float] * maxItr -> Policy
    policy = Whittle(arrival_rates, service_rates, holding_cost_rates) #Initial policy currently Whittle
    itr = 0
    job_classes = [simulations.JobClass(n+1, arrival_rates[n], lib.exp(service_rates[n]))
                   for n in range(len(arrival_rates))]
    
    #maxItr iterations of updating the policy with gittins
    while(itr < maxItr):
        #Simulating with current policy
        itr += 1
        simulated_MG1 = simulations.MG1(job_classes, policy)    
        simulated_MG1.run()
        sim_res = [[job['response_time'] for job in simulated_MG1.metrics
                    if job['job_class'] == n+1] for n in range(len(arrival_rates))]
        
        #Printing simulation results
        print(f"____ Iteration: {itr} ____")
        for n in range(len(arrival_rates)):
            T = sim_res[n]
            print(f"SIMULATED: Class {n+1} mean: {np.mean(T)}, var.: {np.var(T)}")

        #Updating the policy
        policy = nextGittins(service_rates, holding_cost_rates, sim_res, age_values)

    return policy

if __name__ == "__main__":
    l1, l2, mu1, mu2 = 3/8, 3/8, 3, 1
    c1, c2 = lambda t : 10 if t > 50/3 else 0, lambda t : 1
    iterativeGittins([l1, l2], [mu1, mu2], [c1, c2], 10)