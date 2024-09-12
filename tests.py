import math, random
import numpy as np
import derivations, simulations
from termcolor import colored

def print_test(test_name, empirical_val, theoretical_val):
    max_err = 1e-1
    true_err = np.abs(empirical_val - theoretical_val)
    
    if true_err < max_err:
        print(f"PASSED: {test_name}. Value: {theoretical_val}, Error: {true_err}")
    else:
        print(colored(f"FAILED: {test_name}. Theoretical Value: {theoretical_val}, Empirical Value: {empirical_val}", 'red'))

def run_MG1_tests(test_label, l, service_dist, priority_fn = None):
    simulated_MG1 = simulations.MG1([simulations.JobClass(0, l, service_dist, priority_fn)],
                                    is_preemptive=True)
    simulated_MG1.run()
    response_times = [job['response_time'] for job in simulated_MG1.metrics]
    job_sizes = [job['job_size'] for job in simulated_MG1.metrics]
    empirical_mean, empirical_var = np.mean(response_times), np.var(response_times)

    #theoretical_MG1 = derivations.MG1(l, derivations.moments_from_samples(job_sizes))
    theoretical_MG1 = derivations.MG1(l, derivations.moments_from_sample_gen(service_dist))
    E1, ET, ETsq = theoretical_MG1.T_FCFS()
    VarT = ETsq - ET**2

    print_test(test_label + " mean", empirical_mean, ET)
    print_test(test_label + " var", empirical_var, VarT)
                                     
                                                          
def run_NPPrio_tests(test_label, l1, l2, S1, S2):
    simulated_MG1 = simulations.MG1([simulations.JobClass(1, l1, S1),
                                     simulations.JobClass(2, l2, S2)])
    simulated_MG1.run()
    T1 = [job['response_time'] for job in simulated_MG1.metrics if job['job_class'] == 1]
    T2 = [job['response_time'] for job in simulated_MG1.metrics if job['job_class'] == 2]
    empirical_ET1, empirical_VarT1 = np.mean(T1), np.var(T1)
    empirical_ET2, empirical_VarT2 = np.mean(T2), np.var(T2)

    theoretical_MG1 = derivations.TwoClassMG1(l1, l2, #derivations.exp(mu1), derivations.exp(mu2))
                                              derivations.moments_from_sample_gen(S1),
                                              derivations.moments_from_sample_gen(S2))
    T1, T2 = theoretical_MG1.T_NPPrio12()
    E1, ET1, ET1sq = T1
    E1, ET2, ET2sq = T2
    VarT1, VarT2 = ET1sq - ET1**2, ET2sq - ET2**2

    print_test(test_label+" NPPrio Class 1 mean", empirical_ET1, ET1)
    print_test(test_label+" NPPrio Class 1 var", empirical_VarT1, VarT1)
    print_test(test_label+" NPPrio Class 2 mean", empirical_ET2, ET2)
    print_test(test_label+" NPPrio Class 2 var", empirical_VarT2, VarT2)

def run_PPrio_tests(test_label, l1, l2, S1, S2):
    simulated_MG1 = simulations.MG1([simulations.JobClass(1, l1, S1),
                                     simulations.JobClass(2, l2, S2)], is_preemptive = True)
    simulated_MG1.run()
    T1 = [job['response_time'] for job in simulated_MG1.metrics if job['job_class'] == 1]
    T2 = [job['response_time'] for job in simulated_MG1.metrics if job['job_class'] == 2]

    empirical_ET1, empirical_VarT1 = np.mean(T1), np.var(T1)
    empirical_ET2, empirical_VarT2 = np.mean(T2), np.var(T2)

    theoretical_MG1 = derivations.TwoClassMG1(l1, l2, #derivations.exp(mu1), derivations.exp(mu2))
                                              derivations.moments_from_sample_gen(S1),
                                              derivations.moments_from_sample_gen(S2))
    T1, T2 = theoretical_MG1.T_PPrio12()
    E1, ET1, ET1sq = T1
    E1, ET2, ET2sq = T2
    VarT1, VarT2 = ET1sq - ET1**2, ET2sq - ET2**2

    print_test(test_label+" PPrio Class 1 mean", empirical_ET1, ET1)
    print_test(test_label+" PPrio Class 1 var", empirical_VarT1, VarT1)
    print_test(test_label+" PPrio Class 2 mean", empirical_ET2, ET2)
    print_test(test_label+" PPrio Class 2 var", empirical_VarT2, VarT2)    

# random variable generators
def hyperexponential(mu, Csq):
    p = 0.5 * (1 + math.sqrt((Csq - 1)/ (Csq + 1)))
    mu1, mu2 = 2 * p * mu, 2 * (1-p) * mu
    def gen():
        if np.random.uniform() < p:
            return random.expovariate(mu1)
        else:
            return random.expovariate(mu2)
    return gen

def pareto(k = 1965.5, p = 10**10, a = 2.9): # mu = 1/3
    return lambda : 0.001 * (k**(-a) - (1-(k/p)**a)/k**a * np.random.uniform()) ** (-1/a)
    
if __name__ == "__main__":
    print("**MG1 FCFS TESTS**")
    l, mu = 1, 3
    run_MG1_tests("MM1", l, lambda:random.expovariate(mu))
    run_MG1_tests("MD1", l, lambda:1/mu)
    run_MG1_tests("MH1",l,hyperexponential(mu,Csq=5))
    run_MG1_tests("MPar1",1/9,pareto())

    print("**2 CLASS NPPRIO TESTS**")
    l1, l2, mu1, mu2 = 0.8, 0.3, 3, 1
    run_NPPrio_tests("2cMM1a",l1,l2,lambda:random.expovariate(mu1),lambda:random.expovariate(mu2))
    run_NPPrio_tests("2cMM1b",l2,l1,lambda:random.expovariate(mu2),lambda:random.expovariate(mu1))
    run_NPPrio_tests("2cMD1a",l1,l2,lambda:1/mu1,lambda:1/mu2)
    run_NPPrio_tests("2cMD1b",l2,l1,lambda:1/mu2,lambda:1/mu1)
    run_NPPrio_tests("M/DM/1",l1,l2,lambda:1/mu1,lambda:random.expovariate(mu2))
    run_NPPrio_tests("M/MD/1",l1,l2,lambda:random.expovariate(mu1),lambda:1/mu2)
    run_NPPrio_tests("MH1",l1,l2,hyperexponential(mu1,Csq=5),hyperexponential(mu2,Csq=7))

    print("**2 CLASS PPRIO TESTS**")
    run_PPrio_tests("2cMM1c",l1,l2,lambda:random.expovariate(mu1),lambda:random.expovariate(mu2))
    run_PPrio_tests("2cMM1d",l2,l1,lambda:random.expovariate(mu2),lambda:random.expovariate(mu1))
    run_PPrio_tests("2cMD1c",l1,l2,lambda:1/mu1,lambda:1/mu2)
    run_PPrio_tests("2cMD1d",l2,l1,lambda:1/mu2,lambda:1/mu1)
    run_PPrio_tests("2cMH1c",l1,l2,hyperexponential(mu1,Csq=5),hyperexponential(mu2,Csq=5))
    run_PPrio_tests("2cMH1d",l2,l1,hyperexponential(mu2,Csq=10),hyperexponential(mu1,Csq=10))

    print("**2 CLASS NP-ACCPRIO TESTS**")
    run_NPPrio_tests("2cMM1e",l1,l2,lambda:random.expovariate(mu1),lambda:random.expovariate(mu2))
    print("**MG1 PREEMPTIVE POLICIES**")
    l, mu = 1, 3
    #run_MG1_tests("MD1", l, lambda:random.expovariate(mu), lambda x:-x[0])
    
