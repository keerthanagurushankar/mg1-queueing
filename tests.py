import math, random
import numpy as np
import derivations, simulations, lib
from termcolor import colored

def print_test(test_label, empirical_val, theoretical_val):
    max_err = 1e-1
    true_err = np.abs(empirical_val - theoretical_val)
    
    if true_err < max_err:
        print(f"PASSED: {test_label}. Value: {empirical_val}, Error: {true_err}")
    else:
        print(colored(f"FAILED: {test_label}. Theoretical Value: {theoretical_val}, Empirical Value: {empirical_val}", 'red'))

def run_MG1_tests(test_label, l, service_dist, priority_fn=None):
    simulated_MG1 = simulations.MG1([simulations.JobClass(0, l, service_dist, priority_fn)])
    simulated_MG1.run()
    response_times = [job['response_time'] for job in simulated_MG1.metrics]
    job_sizes = [job['job_size'] for job in simulated_MG1.metrics]
    empirical_mean, empirical_var = np.mean(response_times), np.var(response_times)

    theoretical_MG1 = derivations.MG1(l, lib.moments_from_sample_gen(service_dist))
    E1, ET, ETsq = theoretical_MG1.T_FCFS()
    VarT = ETsq - ET**2

    print_test(test_label + " mean", empirical_mean, ET)
    print_test(test_label + " var", empirical_var, VarT)

def run_2Class_MG1_tests(test_label, simulated_MG1, theoretical_T):
    simulated_MG1.run()
    T1 = [job['response_time'] for job in simulated_MG1.metrics if job['job_class'] == 1]
    T2 = [job['response_time'] for job in simulated_MG1.metrics if job['job_class'] == 2]
    empirical_ET1, empirical_VarT1 = np.mean(T1), np.var(T1)
    empirical_ET2, empirical_VarT2 = np.mean(T2), np.var(T2)

    T1, T2 = theoretical_T
    E1, ET1, ET1sq = T1
    E1, ET2, ET2sq = T2
    VarT1, VarT2 = ET1sq - ET1**2, ET2sq - ET2**2

    print_test(test_label+" Class 1 mean", empirical_ET1, ET1)
    print_test(test_label+" Class 1 var", empirical_VarT1, VarT1)
    print_test(test_label+" Class 2 mean", empirical_ET2, ET2)
    print_test(test_label+" Class 2 var", empirical_VarT2, VarT2)    
                                     
                                                          
def run_NPPrio_tests(test_label, l1, l2, S1, S2, V1=None, V2=None):
    simulated_MG1 = simulations.MG1([simulations.JobClass(1, l1, S1, V1),
                                     simulations.JobClass(2, l2, S2, V2)],
                                    is_dynamic_priority = (V1 != None))
    theoretical_MG1 = derivations.TwoClassMG1(l1,l2,
                                              lib.moments_from_sample_gen(S1),
                                              lib.moments_from_sample_gen(S2))
    run_2Class_MG1_tests(test_label+" NPPrio", simulated_MG1, theoretical_MG1.T_NPPrio12())
    
def run_PPrio_tests(test_label, l1, l2, S1, S2):
    simulated_MG1 = simulations.MG1([simulations.JobClass(1, l1, S1),
                                     simulations.JobClass(2, l2, S2)], is_preemptive = True)
    theoretical_MG1 = derivations.TwoClassMG1(l1, l2,
                                              lib.moments_from_sample_gen(S1),
                                              lib.moments_from_sample_gen(S2))
    run_2Class_MG1_tests(test_label+" PPrio", simulated_MG1, theoretical_MG1.T_PPrio12())
    
def run_NPAccPrio_FCFS_tests(test_label, l1, l2, S1, S2):
    simulated_MG1 = simulations.MG1([simulations.JobClass(1, l1, S1, lambda x:x[2]),
                                     simulations.JobClass(2, l2, S2, lambda x:x[2])],
                                    is_dynamic_priority = True)
    theoretical_MG1 = derivations.TwoClassMG1(l1, l2,
                                              lib.moments_from_sample_gen(S1),
                                              lib.moments_from_sample_gen(S2))
    run_2Class_MG1_tests(test_label+" FCFS", simulated_MG1, theoretical_MG1.T_FCFS())


if __name__ == "__main__":
    # print("**MG1 FCFS TESTS**")
    # l, mu = 1, 3
    # run_MG1_tests("MM1", l, lambda:random.expovariate(mu))
    # run_MG1_tests("MD1", l, lambda:1/mu)
    # run_MG1_tests("MH1", l, lib.hyperexponential(mu,Csq=5))
    # run_MG1_tests("MPar1",l,lib.pareto(mu))

    # print("**2 CLASS NPPRIO TESTS**")
    l1, l2, mu1, mu2 = 4, 0.3, 10, 1
    # run_NPPrio_tests("2cMM1a",l1,l2,lambda:random.expovariate(mu1),lambda:random.expovariate(mu2))
    # run_NPPrio_tests("2cMM1b",l2,l1,lambda:random.expovariate(mu2),lambda:random.expovariate(mu1))
    # run_NPPrio_tests("2cMD1a",l1,l2,lambda:1/mu1,lambda:1/mu2)
    # run_NPPrio_tests("2cMD1b",l2,l1,lambda:1/mu2,lambda:1/mu1)
    # run_NPPrio_tests("M/DM/1",l1,l2,lambda:1/mu1,lambda:random.expovariate(mu2))
    # run_NPPrio_tests("M/MD/1",l1,l2,lambda:random.expovariate(mu1),lambda:1/mu2)
    # run_NPPrio_tests("2cMH1a",l1,l2,lib.hyperexponential(mu1,Csq=5),lib.hyperexponential(mu2,Csq=7))

    # print("**2 CLASS PPRIO TESTS**")
    # run_PPrio_tests("2cMM1c",l1,l2,lambda:random.expovariate(mu1),lambda:random.expovariate(mu2))
    # run_PPrio_tests("2cMM1d",l2,l1,lambda:random.expovariate(mu2),lambda:random.expovariate(mu1))
    # run_PPrio_tests("2cMD1c",l1,l2,lambda:1/mu1,lambda:1/mu2)
    # run_PPrio_tests("2cMD1d",l2,l1,lambda:1/mu2,lambda:1/mu1)
    # run_PPrio_tests("2cMH1c",l1,l2,lib.hyperexponential(mu1,Csq=5),lib.hyperexponential(mu2,Csq=5))
    # run_PPrio_tests("2cMH1d",l2,l1,lib.hyperexponential(mu2,Csq=10),lib.hyperexponential(mu1,Csq=10))
    
    print("**2 CLASS NP-ACC-PRIO TESTS**")
    b1, b2 = 100, 1 # should be close to NPPrio12
    run_NPPrio_tests("2cMM1acc100", l1, l2,
                     lambda:random.expovariate(mu1), lambda:random.expovariate(mu2), 
                     lambda x:b1*x[2], lambda x:b2*x[2])
    run_NPPrio_tests("2cMD1acc100", l1, l2,
                     lambda:1/mu1, lambda:1/mu2,
                     lambda x:b1*x[2], lambda x:b2*x[2])
    run_NPPrio_tests("2cMH1acc100", l1, l2,
                      lib.hyperexponential(mu1,Csq=5), lib.hyperexponential(mu2,Csq=7),
                      lambda x:b1*x[2], lambda x:b2*x[2])   
    b1, b2 = 1, 1 # should be FCFS
    run_NPAccPrio_FCFS_tests("2cMM1acc1", l1, l2,
                              lambda:random.expovariate(mu1), lambda:random.expovariate(mu2))
    run_NPAccPrio_FCFS_tests("2cMD1acc1", l1, l2,
                              lambda:1/mu1, lambda:1/mu2)
    run_NPAccPrio_FCFS_tests("2cMH1acc1", l1, l2,
                              lib.hyperexponential(mu1,Csq=5), lib.hyperexponential(mu2,Csq=7))
    
