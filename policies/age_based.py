from .base import Policy
import random, numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import logging

# delay based preemptive index priority policy
def AgeBasedPrio(V, age_values=np.arange(10, 20, 0.1), num_classes=2):
    # list of floats, list of floats, list of fns, np.array
    V_values = [[V(0, 0, t, k) for t in age_values] for k in range(1, num_classes+1)]
    #V_values = [[Vi(0, 0, t) for t in age_values] for Vi in prio_fns]

    # if job2 has currently lower prio, but may be higher later, compute when.    
    def calculate_overtake_time(job1, job2, current_time):
        class1, t1 = job1.job_class.index-1, job1.arrival_time
        class2, t2 = job2.job_class.index-1, job2.arrival_time

        if class1 == class2:
            return None

        V1 = lambda t: np.interp(t-t1, age_values, V_values[class1])
        V2 = lambda t: np.interp(t-t2, age_values, V_values[class2])
        overtake_cond = lambda t: V2(t) - V1(t)
        max_time = t2 + age_values[-1]
        
        logging.debug(f"Checking for overtake of {class2+1, t2} over {class1+1, t1}")
        
        if overtake_cond(current_time) < 0 and overtake_cond(max_time) > 0:
            overtake_time = opt.brentq(overtake_cond, current_time, max_time)
            return overtake_time + 0.005 if overtake_time > current_time else None

        if overtake_cond(current_time) < 0 and overtake_cond(max_time) < 0:
            for t in np.linspace(current_time, max_time, 5)[1:-1]:
                if overtake_cond(t) > 0:
                    overtake_time = opt.brentq(overtake_cond, current_time, t)
                    return overtake_time + 0.005 if overtake_time > current_time else None

        return None    

    policy = Policy("AgeBased", priority_fn = V,  is_preemptive=True,
                  is_dynamic_priority=True,
                  calculate_overtake_time=calculate_overtake_time)
    policy.priority_values = V_values
    return policy

def generalized_cmu(service_rates, holding_cost_rates, age_values=np.arange(0, 15, 0.1)):
    prio_fns = [lambda r, s, t : mu*c(t) for mu,c in zip(service_rates, holding_cost_rates)]
    prio_fn = lambda r, s, t, k: service_rates[k-1] * holding_cost_rates[k-1](t)
    policy = AgeBasedPrio(prio_fn, age_values, num_classes=len(service_rates))
    policy.policy_name = r'gen-$c\mu$'
    return policy
    
def Whittle(arrival_rates, service_rates, holding_cost_rates, 
            age_values=np.arange(0, 15, 0.1)):
    # list[float] * list[float] * list[float -> float] -> Policy
    # Vi(t) = mui * E[ci(t + T)] where T ~ Exp(mui - li)
    V_values = []
    for l, mu, c in zip(arrival_rates, service_rates, holding_cost_rates):
        Ti = [random.expovariate(mu - l) for _ in range(10**4)]
        Vi_values = np.array([mu * np.mean([c(t + T) for T in Ti]) for t in age_values])
        V_values.append(Vi_values)
    
    V = lambda r, s, t, k : np.interp(t, age_values, V_values[k-1])
    policy = AgeBasedPrio(V, age_values, num_classes=len(service_rates))
    policy.policy_name = "Whittle"
    return policy

def Aalto(arrival_rates, service_rates, holding_cost_rates, 
            age_values=np.arange(0, 15, 0.1)):
    # list[float] * list[float] * list[float -> float] -> Policy
    # Vi(t) = mui * E[ci(t + S)] where S ~ Exp(mui)
    V_values = []
    for l, mu, c in zip(arrival_rates, service_rates, holding_cost_rates):
        Si = [random.expovariate(mu ) for _ in range(10**4)]
        Vi_values = np.array([mu * np.mean([c(t + S) for S in Si]) for t in age_values])
        V_values.append(Vi_values)
    
    V = lambda r, s, t, k : np.interp(t, age_values, V_values[k-1])
    policy = AgeBasedPrio(V, age_values, num_classes=len(service_rates))
    policy.policy_name = "Aalto"
    return policy
