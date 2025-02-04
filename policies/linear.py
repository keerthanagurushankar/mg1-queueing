from .base import Policy
import logging

def LinearAccPrio(a_values, b_values, is_preemptive=True):
    V = lambda r, s, t, k: a_values[k-1] + b_values[k-1] * t
    policy_name = ("PAAPQ" if is_preemptive else "NPAAPQ")\
        + f"({a_values, b_values})" 

    def calculate_overtake_time(job1, job2, current_time):
        i1, i2 = job1.job_class.index-1, job2.job_class.index-1
        a1, b1, t1 = a_values[i1], b_values[i1], job1.arrival_time
        a2, b2, t2 = a_values[i2], b_values[i2], job2.arrival_time

        if b2 <= b1:
            return None
        overtake_time = (a1 - a2 + b2 * t2 - b1 * t1) / (b2 - b1)
        logging.debug(f"Checking for overtake of {i2+1, t2} over {i1+1, t1} at {overtake_time}")
        return overtake_time + 0.001 if overtake_time >= current_time else None  

    return Policy(policy_name, priority_fn=V, is_preemptive=is_preemptive,
                  is_dynamic_priority=True, calculate_overtake_time=calculate_overtake_time)

def AccPrio(b1, b2, is_preemptive=True):
    policy = LinearAccPrio([0, 0], [b1, b2], is_preemptive)
    policy.policy_name = ("" if is_preemptive else "N") + f"PAccPrio({round(b1, 2)}, {round(b2, 2)})"
    return policy

def Lookahead(alpha):
    policy = LinearAccPrio([0, alpha], [1, 0], is_preemptive=True)
    policy.policy_name = f"Lookahead({round(alpha, 2)})"
    return policy

def LinearWhittle(arrival_rates, service_rates, cost_rates):
    # list of floats, list of floats, list of pairs of floats
    # whittle policy if inst ci(t) = ci * t + di, whittle idx is an affine priority
    # Vi(t) = mui * E[ci * (t + ExpT) + di] = mui ci t + mui ci / (mui- li) + mui * di    
    a_values, b_values = [], []
    for li, mui, (ci, di) in zip(arrival_rates, service_rates, cost_rates):
        a_values.append(mui * ci / (mui - li) + mui * di)
        b_values.append(mui * ci)

    policy = LinearAccPrio(a_values, b_values, is_preemptive=True)
    policy.policy_name = "Whittle"
    return policy

def LinearAalto(arrival_rates, service_rates, cost_rates):
    # Vi(t) = mui * E[ci * (t +  Exp(mui)) + di]
    a_values, b_values = [], []
    for li, mui, (ci, di) in zip(arrival_rates, service_rates, cost_rates):
        a_values.append(mui * ci / mui + mui * di)
        b_values.append(mui * ci)

    policy = LinearAccPrio(a_values, b_values, is_preemptive=True)
    policy.policy_name = "Aalto"
    return policy    
