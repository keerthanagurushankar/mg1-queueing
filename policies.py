class Policy:
    def __init__(self, policy_name, priority_fn=None, is_preemptive=False,
                 is_dynamic_priority=False, calculate_overtake_time=lambda j1,j2,t:None):
        self.policy_name = policy_name
        self.priority_fn = priority_fn
        self.is_preemptive = is_preemptive
        self.is_dynamic_priority = is_dynamic_priority
        self.calculate_overtake_time = calculate_overtake_time

# SIMPLE POLICIES

FCFS = Policy("FCFS", priority_fn=lambda r, s, t, k: t, is_preemptive=False)
SRPT = Policy("SRPT", priority_fn=lambda r, s, t, k:-r, is_preemptive=True, is_dynamic_priority = True)

# CLASS BASED POLICIES

NPPrio12 = Policy("NPPrio12", priority_fn=lambda r, s, t, k:-k, is_preemptive=False)
PPrio12 = Policy("PPrio12", priority_fn=lambda r, s, t, k:-k, is_preemptive=True)
NPPrio21 = Policy("NPPrio21", priority_fn=lambda r, s, t, k:k, is_preemptive=False)
PPrio21 = Policy("PPrio21", priority_fn=lambda r, s, t, k:k, is_preemptive=True)

def LinearAccPrio(a_values, b_values, is_preemptive=True):
    # Vi(t) = ai + bi t

    V = lambda r, s, t, k: a_values[k-1] + b_values[k-1] * t 
    policy_name = "PAAPQ" if is_preemptive else "NPAAPQ"

    def calculate_overtake_time(job1, job2, current_time=None):
        i1, i2 = job1.job_class.index-1, job2.job_class.index-1
        a1, b1, t1 = a_values[i1], b_values[i1], job1.arrival_time 
        a2, b2, t2 = a_values[i2], b_values[i2], job2.arrival_time
        # a1 + b1 (t - t1) = a2 + b2 (t - t2) gives t : time of overtake
        if b2 <= b1:
            return None
        overtake_time = (a1 - a2 + b2 * t2 - b1 * t1) / (b2 - b1)
        logging.debug(f"Checking for overtake of {i2+1, t2} over {i1+1, t1}"
                      f"at {overtake_time}")
        return overtake_time + 0.001 if overtake_time >= current_time else None  

    return Policy((policy_name, a_values, b_values), priority_fn=V,
                  is_preemptive=is_preemptive,
                  is_dynamic_priority=True,
                  calculate_overtake_time=calculate_overtake_time) 

def AccPrio(bs, is_preemptive=True):
    return LinearAccPrio([0]*len(bs), bs, is_preemptive)

def AccPrio(b1, b2, is_preemptive=True):
    policy = LinearAccPrio([0, 0], [b1, b2], is_preemptive)
    policy.policy_name = ("" if is_preemptive else "N") + "PAccPrio(" \
        + str(round(b1, 2)) + ", " + str(round(b2, 2)) + ")"
    return policy

def Lookahead(alpha):
    policy = LinearAccPrio([0, alpha], [1, 0], is_preemptive=True)
    policy.policy_name = f"Lookahead({round(alpha, 2)})"
    return policy

def QuadraticAccPrio(a_values, b_values, c_values, is_preemptive=True):
    # Vi(t : age) = ai t^2 + bi t + ci
    V = lambda r, s, t, k: a_values[k-1] * t**2 + b_values[k-1] * t + c_values[k-1]
    policy_name = ("" if is_preemptive else "N") +  "PQAPQ" + \
        "(" + str(a_values) + ")" #b_values, c_values
    
    def calculate_overtake_time(job1, job2, current_time):
        i1, i2 = job1.job_class.index-1, job2.job_class.index-1
        if i1 == i2:
            return None
        
        a1, b1, c1, t1 = a_values[i1], b_values[i1], c_values[i1], job1.arrival_time
        a2, b2, c2, t2 = a_values[i2], b_values[i2], c_values[i2], job2.arrival_time
        # Vi(t : time) = ai (t-ti)^2 + bi(t-ti) + ci
        # a1 (t - t1)^2 + b1 (t - t1) + c1 = a2 (t - t2) ^ 2 + b2 (t - t2) + c2
        # a1 t^2 - 2a1 t1 t + a1 t1^2 + b1 t - b1 t1 + c1
        # = a2 t^2 - 2a2 t2 t + a2 t2^2 + b2 t - b2 t2 + c2
        # Given V1(t0) > V2(t0), want smallest t s.t. V1(t) <= V2(t)
        # Given P(t) = At**2 + Bt + C > 0 now, want smallest 
        A = - a1 + a2
        B = 2 * a1 * t1 - b1 - 2 * a2 * t2 + b2
        C = - a1 * t1**2 + b1 * t1 - c1 + a2 * t2**2 - b2 * t2 + c2
        D = B**2 - 4 * A * C
        # overtake_cond = lambda t : V2(t) - V1(t)
        overtake_cond = lambda t : A * t**2 + B * t + C

        #assert overtake_cond(current_time) <= 0.01, "Only check overtake of lagging job"
        


        logging.debug(f"Checking for overtake of {i2+1, t2} over {i1+1, t1} ")
                   #   f"at { (-B + np.sqrt(D)) / (2 * A) }")

        if D <= 0:
            logging.debug(f"quadratic fn has no root")
            return None
            
        if A > 0:
            overtake_time = (-B + np.sqrt(D)) / (2 * A)
            logging.debug(f"Found one for {overtake_time}")
            return overtake_time + 0.001 if overtake_time >= current_time else None
        elif A < 0: # overtake_cond(current_time) >= 0:
            logging.debug(f"{current_time, (-B-np.sqrt(D)) / (2*A)}")
            overtake_time = max(current_time, (-B+np.sqrt(D)) / (2*A)) 
            logging.debug(f"Found one for {overtake_time}")

            # debug
            # if i2 == 1 and i1 == 0:
            #     age_values = np.arange(0, 50, 0.5)
            #     plt.plot(age_values, [overtake_cond(t+t2) for t in age_values])
            #     plt.plot(age_values, np.zeros(len(age_values)))
            #     plt.axvline(x=overtake_time)
            #     plt.show()
                
            return overtake_time + 0.001           
        else:
            return None
        
    return Policy(policy_name, V, is_preemptive, is_dynamic_priority=True,
                  calculate_overtake_time=calculate_overtake_time)
        

# AGE VARYING INDEX POLICIES

import random, numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import logging

# delay based preemptive index priority policy
def AgeBasedPrio(V, age_values=np.arange(10, 20, 0.1)):
    # list of floats, list of floats, list of fns, np.array
    V_values = [[V(0, 0, t, 1) for t in age_values], [V(0, 0, t, 2) for t in age_values]]
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
    policy = AgeBasedPrio(prio_fn, age_values)
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
         #   for k in range(len(arrival_rates))]
    policy = AgeBasedPrio(V, age_values)
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
         #   for k in range(len(arrival_rates))]
    policy = AgeBasedPrio(V, age_values)
    policy.policy_name = "Whittle"
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

def QuadraticWhittle(arrival_rates, service_rates, cost_rates):
    # if ci(t) = a t^2 + bt + c then
    # Vi(t) / mui = E[a(t + T)^2 + b(t + T) + c]
    # = at^2 + 2at/(mu-l) + bt + a * 2/(mu-l)**2 + b/(mu-l) + c
    a_values, b_values, c_values = [], [], []

    for l, mu, (a, b, c) in zip(arrival_rates, service_rates, cost_rates):
        a_values.append(mu * a)
        b_values.append(mu * (2 * a / (mu - l) + b))
        c_values.append(mu * (a * 2 / (mu - l)**2 + b / (mu - l) + c))

    policy = QuadraticAccPrio(a_values, b_values, c_values, is_preemptive=True)
    policy.policy_name = "Whittle"
    return policy

def QuadraticAalto(arrival_rates, service_rates, cost_rates):
    # if ci(t) = a t^2 + bt + c then
    # Vi(t) / mui = E[a(t + S)^2 + b(t + S) + c]
    # = at^2 + 2at/(mu) + bt + a * 2/(mu)**2 + b/(mu) + c
    a_values, b_values, c_values = [], [], []

    for l, mu, (a, b, c) in zip(arrival_rates, service_rates, cost_rates):
        a_values.append(mu * a)
        b_values.append(mu * (2 * a / (mu) + b))
        c_values.append(mu * (a * 2 / (mu)**2 + b / (mu) + c))

    policy = QuadraticAccPrio(a_values, b_values, c_values, is_preemptive=True)
    policy.policy_name = "Aalto"
    return policy
