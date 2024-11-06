class Policy:
    def __init__(self, policy_name, priority_fn=None, is_preemptive=False,
                 is_dynamic_priority=False, calculate_overtake_time=lambda j1,j2,t:None):
        self.policy_name = policy_name
        self.priority_fn = priority_fn
        self.is_preemptive = is_preemptive
        self.is_dynamic_priority = is_dynamic_priority
        self.calculate_overtake_time = calculate_overtake_time

# SIMPLE POLICIES

FCFS = Policy("FCFS")
SRPT = Policy("SRPT", priority_fn=lambda r, s, t, k:-r, is_preemptive=True, is_dynamic_priority = True)

V1, V2 = lambda r, s, t:-1, lambda r, s, t:-2
NPPrio12 = Policy("NPPrio12", priority_fn=[V1, V2], is_preemptive=False)
PPrio12 = Policy("PPrio12", priority_fn=[V1, V2], is_preemptive=True)

V1, V2 = lambda r, s, t:-2, lambda r, s, t:-1
NPPrio21 = Policy("NPPrio21", priority_fn=[V1, V2], is_preemptive=False)
PPrio21 = Policy("PPrio21", priority_fn=[V1, V2], is_preemptive=True)

def AccPrio(b1, b2, is_preemptive=False):
    V1 = lambda r, s, t: b1 * t
    V2 = lambda r, s, t: b2 * t
    policy_name = "PAccPrio" if is_preemptive else "NPAccPrio"

    b = {1 : b1, 2: b2}
    def calculate_overtake_time(job1, job2, current_time=None):
        preemption_delay = 0.001
        b1, t1 = b[job1.job_class.index], job1.arrival_time 
        b2, t2 = b[job2.job_class.index], job2.arrival_time
        # b1 (t - t1) = b2 (t - t2) gives t : further time to overtake
        if b2 <= b1:
            return None
        return (b2 * t2 - b1 * t1)/(b2-b1) + preemption_delay
        
    return Policy((policy_name, b1, b2), priority_fn=[V1, V2],
                  is_preemptive=is_preemptive,
                  is_dynamic_priority=True,
                  calculate_overtake_time=calculate_overtake_time)

def Lookahead(alpha):
    V1 = lambda r, s, t: t
    V2 = lambda r, s, t: alpha

    # Vi(t) = ai + bi * t
    a = {1: 0, 2: alpha}
    b = {1: 1, 2: 0}
    
    def calculate_overtake_time(job1, job2, current_time=None):
        preemption_delay = 0.001
        a1, b1, t1 = a[job1.job_class.index], b[job1.job_class.index], job1.arrival_time
        a2, b2, t2 = a[job2.job_class.index], b[job2.job_class.index], job2.arrival_time
        # a1 + b1 (t - t1) = a2 + b2 (t - t2)
        if b2 <= b1:
            return None
        return (a1 - a2 + b2 * t2 - b1 * t1) / (b2 - b1) + preemption_delay
    
    return Policy(("Lookahead", alpha), priority_fn=[V1, V2],
                  is_preemptive=True, is_dynamic_priority=True,
                  calculate_overtake_time = calculate_overtake_time)

# AGE VARYING INDEX POLICIES

import random, numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)

# delay based whittle index priority for k-class M/M/1
def Whittle(arrival_rates, service_rates, holding_cost_rates, age_values=None):
    if not age_values:
        age_values = np.arange(0, 15, 0.1)

    # compute priority functions
    V_values = []
    for l, mu, c in zip(arrival_rates, service_rates, holding_cost_rates):
        Ti = [random.expovariate(mu - l) for _ in range(10**4)]
        Vi_values = np.array([mu * np.mean([c(t + T) for T in Ti]) for t in age_values])
        V_values.append(Vi_values)
    V = lambda r, s, t, k: np.interp(t, age_values, V_values[k-1])

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
                    return overtake_time + 0.05 if overtake_time > current_time else None

        return None    
    
    return Policy("Whittle", priority_fn=V, is_preemptive=True,
                  is_dynamic_priority=True,
                  calculate_overtake_time=calculate_overtake_time)

if __name__ == "__main__":
    l1, l2, mu1, mu2 = .4, .5, 1, 1.5
    c1, c2 = lambda t : 2 if t > 10 else 0, lambda t : 1
    #WhittleIdx = Whittle([l1, l2], [mu1, mu2], [c1, c2])    
    c1, c2 = lambda t : 2 if t > 10 else 0, lambda t : 1 if t > 5 else 0
    WhittleIdx = Whittle([l1, l2], [mu1, mu2], [c1, c2])
    

        # if class2 == 1 and class1 == 0:
        #     plt.plot(age_values, [overtake_cond(t+t2) for t in age_values])
        #     plt.plot(age_values, np.zeros(len(age_values)))
        #     plt.show()
