from .base import Policy
import numpy as np
import logging
import matplotlib.pyplot as plt

def QuadraticAccPrio(a_values, b_values, c_values, is_preemptive=True):
    # Vi(t) = a_i * t^2 + b_i * t + c_i
    #    a_values (list of floats): Coefficients for quadratic term per class.
    #    b_values (list of floats): Coefficients for linear term per class.
    #    c_values (list of floats): Constant term per class.
    
    V = lambda r, s, t, k: a_values[k-1] * t**2 + b_values[k-1] * t + c_values[k-1]
    policy_name = ("PQAPQ" if is_preemptive else "NPAAPQ") + f"({a_values})"

    def calculate_overtake_time(job1, job2, current_time):
        # Computes the earliest time when job2 overtakes job1 in priority.
        
        i1, i2 = job1.job_class.index - 1, job2.job_class.index - 1
        if i1 == i2:
            return None  # Same class jobs won't overtake each other.

        a1, b1, c1, t1 = a_values[i1], b_values[i1], c_values[i1], job1.arrival_time
        a2, b2, c2, t2 = a_values[i2], b_values[i2], c_values[i2], job2.arrival_time

        # Solve for t in:
        # a1(t - t1)^2 + b1(t - t1) + c1 = a2(t - t2)^2 + b2(t - t2) + c2
        # Expanding both sides and simplifying:
        A = -a1 + a2
        B = 2 * a1 * t1 - b1 - 2 * a2 * t2 + b2
        C = -a1 * t1**2 + b1 * t1 - c1 + a2 * t2**2 - b2 * t2 + c2
        D = B**2 - 4 * A * C  # Discriminant

        logging.debug(f"Checking for overtake of {i2+1, t2} over {i1+1, t1}")

        if D < 0:
            logging.debug("Quadratic equation has no real root; no overtake.")
            return None  # No real root means no overtake occurs.

        if A == 0:  # Linear case: Solve Bt + C = 0
            if B != 0:
                overtake_time = -C / B
                return overtake_time + 0.001 if overtake_time >= current_time else None
            else:
                return None  # No valid overtake time if B = 0 and A = 0.

        # Quadratic solutions: t = (-B ± sqrt(D)) / (2A)
        root1 = (-B + np.sqrt(D)) / (2 * A)
        root2 = (-B - np.sqrt(D)) / (2 * A)

        # Select the first valid overtake time that happens in the future
        valid_roots = [r for r in (root1, root2) if r >= current_time]
        if not valid_roots:
            return None

        overtake_time = min(valid_roots) + 0.001
        logging.debug(f"Overtake occurs at {overtake_time}")

        # debug
        overtake_cond = lambda t : A * t**2 + B * t + C
        # if i2 == 1 and i1 == 0:
        #     age_values = np.arange(0, 50, 0.5)
        #     plt.plot(age_values, [overtake_cond(t+t2) for t in age_values])
        #     plt.plot(age_values, np.zeros(len(age_values)))
        #     plt.axvline(x=overtake_time)
            # plt.show()
        
        return overtake_time

    return Policy(policy_name, priority_fn=V, is_preemptive=is_preemptive,
                  is_dynamic_priority=True,
                  calculate_overtake_time=calculate_overtake_time)

def QuadraticGenCMU(service_rates, cost_rates):
    # list[float] * list[float * float * float] -> policy
    a_values, b_values, c_values = [], [], []

    for mu, (a, b, c) in zip(service_rates, cost_rates):
        a_values.append(mu * a)
        b_values.append(mu * b)
        c_values.append(mu * c)

    policy = QuadraticAccPrio(a_values, b_values, c_values, is_preemptive=True)
    policy.policy_name = r"gen-$c\mu$"
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
