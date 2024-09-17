import math, random, numpy as np
import matplotlib.pyplot as plt
import lib, simulations, policies as policy

def compute_PAccPrio(l1, l2, S1, S2, b1, b2):
    PAccPrio = policy.AccPrio(b1, b2, is_preemptive = True)
    simulated_MG1 = simulations.MG1([simulations.JobClass(1, l1, S1), 
                                     simulations.JobClass(2, l2, S2)], PAccPrio)
    simulated_MG1.run()
    response_times = [job['response_time'] for job in simulated_MG1.metrics]
    response_times_sq = [t**2 for t in response_times]
    return np.mean(response_times_sq)
    
def plot():
    l, mu1, mu2 = 1, 2, 3
    S1, S2 = lib.exp(mu1), lib.exp(mu2)
    
    b1s, ETsqs = [], []
    for i in range(3, 10):
        b1, b2 = np.exp(i), 1
        ETsq = compute_PAccPrio(l, l, S1, S2, b1, b2)   
        print(f"Ran computations for {i}, {b1} -> {ETsq}")             
        
        b1s.append(b1)
        ETsqs.append(ETsqs)
        
    plt.plot(b1s, ETsqs)
    plt.xlabel('b1')
    plt.ylabel('ETsq')
    plt.show()
    
if __name__ == "__main__":
    plot()
        
