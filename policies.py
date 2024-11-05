class Policy:
    def __init__(self, policy_name, priority_fn=None, is_preemptive=False,
                 is_dynamic_priority=False, calculate_overtake_time=lambda j1, j2 : None):
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
NPPrio12 = Policy("NPPrio21", priority_fn=[V1, V2], is_preemptive=False)
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

# interpolate f given evaluation at sample values
def interpolated_fx(x, sample_xs, sample_fxs):
    # Use np.interp for linear interpolation
    if x < sample_xs[0]:  # If age is below the sample range, return the first value
        return sample_fxs[0]
    elif x > sample_fxs[-1]:  # If age is above the sample range, return the last value
        return sample_fxs[-1]
    else:
        return np.interp(x, sample_xs, sample_fxs)

# delay based whittle index priority for k-class M/M/1
def Whittle(arrival_rates, service_rates, holding_cost_rates, age_values=None):
    if not age_values:
        age_values = np.arange(0, 20, 0.1)

    # compute priority functions
    V_values = []
    for l, mu, c in zip(arrival_rates, service_rates, holding_cost_rates):
        Ti = [random.expovariate(mu - l) for _ in range(10**4)]
        Vi_values = np.array([mu * np.mean([c(t + T) for T in Ti]) for t in age_values])
        V_values.append(Vi_values)
    V = lambda r, s, t, k: interpolated_fx(t, age_values, V_values[k-1])


    def calculate_overtake_time(job1, job2, current_time):
        class1, t1 = job1.job_class.index-1, job1.arrival_time
        class2, t2 = job2.job_class.index-1, job2.arrival_time
        
        def overtake_cond(age2):
            return np.interp(age2, age_values, V_values[class2]) - \
                np.interp(age2 + (t1-t2), age_values, V_values[class1])
        try:
            curr_age2 = current_time - t2
            overtake_age2 = opt.brentq(overtake_cond, curr_age2, age_values[-1])
            overtake_time = current_time + overtake_age2 - curr_age2
            return overtake_time + 0.3
        except ValueError:
            return None
    
    return Policy("Whittle", priority_fn=V, is_preemptive=True,
                  is_dynamic_priority=True,
                  calculate_overtake_time=calculate_overtake_time)


