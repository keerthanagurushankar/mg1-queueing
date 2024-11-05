import math, random
import numpy as np
import lib, policies as policy, derivations, simulations, indexpolicies
from termcolor import colored

def print_test(test_label, empirical_val, theoretical_val):
    max_err = 1e-2
    true_err = np.abs(empirical_val - theoretical_val) / (empirical_val + theoretical_val)
    
    if true_err < max_err:
        print(f"PASSED: {test_label}. Value: {empirical_val}, Error: {true_err * 100}%")
    else:
        print(colored(f"FAILED: {test_label}. Theoretical Value: {theoretical_val}, Empirical Value: {empirical_val}", 'red'))

def run_MG1_tests(test_label, l, S, policy=policy.FCFS):
    simulated_MG1 = simulations.MG1([simulations.JobClass(0, l, S)], policy)
    simulated_MG1.run()
    response_times = [job['response_time'] for job in simulated_MG1.metrics]
    job_sizes = [job['job_size'] for job in simulated_MG1.metrics]
    empirical_mean, empirical_var = np.mean(response_times), np.var(response_times)

    theoretical_MG1 = derivations.MG1(l, S)
    T = theoretical_MG1.T(policy.policy_name)
    if len(T) >= 2:
        print_test(test_label + " mean", empirical_mean, T[1])
    if len(T) >= 3:
        VarT = T[2] - T[1]**2
        print_test(test_label + " var", empirical_var, VarT)

def run_2Class_MG1_tests(test_label, l1, l2, S1, S2, policy):
    simulated_MG1 = simulations.MG1([simulations.JobClass(1, l1, S1),
                                     simulations.JobClass(2, l2, S2)], policy)    
    simulated_MG1.run()
    T1 = [job['response_time'] for job in simulated_MG1.metrics if job['job_class'] == 1]
    T2 = [job['response_time'] for job in simulated_MG1.metrics if job['job_class'] == 2]
    empirical_ET1, empirical_VarT1 = np.mean(T1), np.var(T1)
    empirical_ET2, empirical_VarT2 = np.mean(T2), np.var(T2)

    theoretical_MG1 = derivations.TwoClassMG1(l1,l2,
                                              lib.moments_from_sample_gen(S1),
                                              lib.moments_from_sample_gen(S2))    
    T = theoretical_MG1.T(policy.policy_name)
    if T is None:
        print(f"SIMULATED: {test_label} Class 1 mean: {empirical_ET1}, var.: {empirical_VarT1}")
        print(f"SIMULATED: {test_label} Class 2 mean: {empirical_ET2}, var.: {empirical_VarT2}")
        return 
    
    T1, T2 = T
    E1, ET1, ET1sq = T1
    E1, ET2, ET2sq = T2
    VarT1, VarT2 = ET1sq - ET1**2, ET2sq - ET2**2

    print_test(test_label+" Class 1 mean", empirical_ET1, ET1)
    print_test(test_label+" Class 1 var", empirical_VarT1, VarT1)
    print_test(test_label+" Class 2 mean", empirical_ET2, ET2)
    print_test(test_label+" Class 2 var", empirical_VarT2, VarT2)

if __name__ == "__main__":
    print("**MG1 FCFS TESTS**")
    l, mu = .4, 2
    # run_MG1_tests("MM1", l, lib.exp(mu))
    # run_MG1_tests("MD1", l, lib.det(mu))
    # run_MG1_tests("MH1", l, lib.hyperexponential(mu,Csq=5))
    # run_MG1_tests("MPar1",l,lib.pareto(mu))

    # print("**2 CLASS NPPRIO TESTS**")
    l1, l2, mu1, mu2 = .4, .3, 1, 1.5
    # run_2Class_MG1_tests("2cMM1a", l1, l2, lib.exp(mu1), lib.exp(mu2), policy.NPPrio12)
    # run_2Class_MG1_tests("2cMM1b", l2, l1, lib.exp(mu2), lib.exp(mu1), policy.NPPrio12)
    # run_2Class_MG1_tests("2cMD1a", l1, l2, lib.det(mu1), lib.det(mu2), policy.NPPrio12)
    # run_2Class_MG1_tests("2cMD1b", l2, l1, lib.det(mu2), lib.det(mu1), policy.NPPrio12)
    # run_2Class_MG1_tests("2cMH1a", l1, l2, lib.hyperexponential(mu1, Csq=5),
    #                     lib.hyperexponential(mu2, Csq=10), policy.NPPrio12)


    print("**2 CLASS PPRIO TESTS**")
    run_2Class_MG1_tests("2cMM1c", l1, l2, lib.exp(mu1), lib.exp(mu2), policy.PPrio12)
    # run_2Class_MG1_tests("2cMM1d", l2, l1, lib.exp(mu2), lib.exp(mu1), policy.PPrio12)
    # run_2Class_MG1_tests("2cMD1c", l1, l2, lib.det(mu1), lib.det(mu2), policy.PPrio12)
    # run_2Class_MG1_tests("2cMH1c", l1, l2, lib.hyperexponential(mu1, Csq=5),
    #                     lib.hyperexponential(mu2, Csq=5), policy.PPrio12)
    # run_2Class_MG1_tests("2cMH1d", l1, l2, lib.hyperexponential(mu1, Csq=10),
    #                     lib.hyperexponential(mu2, Csq=10), policy.PPrio12)
    
    # print("**2 CLASS NP-ACC-PRIO TESTS**")
    # NPAccPrio = policy.AccPrio(b1 = 3, b2 = 2, is_preemptive = False) 
    # run_2Class_MG1_tests("2cMM1NPacc", l1, l2, lib.exp(mu1), lib.exp(mu2), NPAccPrio)
    # run_2Class_MG1_tests("2cMD1NPacc", l1, l2, lib.det(mu1), lib.det(mu2), NPAccPrio)
    # run_2Class_MG1_tests("2cMH1NPacc", l1, l2, lib.hyperexponential(mu1, Csq=5),
    #                     lib.hyperexponential(mu2, Csq=5), NPAccPrio)
    # b1, b2 = 1, 1 

    print("**2 CLASS P-ACC-PRIO TESTS**")
    PAccPrio = policy.AccPrio(b1 = 100, b2 = 1, is_preemptive = True)
    #run_2Class_MG1_tests("2cMM1Pacc", l1, l2, lib.exp(mu1), lib.exp(mu2), PAccPrio)
    # run_2Class_MG1_tests("2cMD1Pacc", l1, l2, lib.det(mu1), lib.det(mu2), PAccPrio)
    # run_2Class_MG1_tests("2cMH1Pacc", l1, l2, lib.hyperexponential(mu1, Csq=5),
    #                      lib.hyperexponential(mu2, Csq=5), PAccPrio)    

    # print("**SRPT TESTS**")
    # l, mu = 7, 10
    # run_MG1_tests("SRPT MM1", l, lib.exp(mu), policy.SRPT)    
    # run_MG1_tests("SRPT MD1", l, lib.det(mu), policy.SRPT)
    # run_MG1_tests("SRPT MH1 Csq10", l, lib.hyperexponential(mu, Csq=10), policy.SRPT)
    # run_MG1_tests("SRPT MH1 Csq50", l, lib.hyperexponential(mu, Csq=50), policy.SRPT)    
    # run_MG1_tests("SRPT MH1 Csq100", l, lib.hyperexponential(mu, Csq=100), policy.SRPT)
    # run_MG1_tests("SRPT MPar1", l, lib.pareto(mu), policy.SRPT)

    print("**LOOKAHEAD TESTS**")
    #run_2Class_MG1_tests("MM1Look0", l1, l2, lib.exp(mu1), lib.exp(mu2), policy.Lookahead(0))
    #run_2Class_MG1_tests("MM1Look10",l1, l2, lib.exp(mu1), lib.exp(mu2), policy.Lookahead(10))

    print("**WHITTLE INDEX TESTS**")
    l1, l2, mu1, mu2 = 0.4, 0.3, 1, 1.5
    c1, c2 = lambda t : 2 if t > 13 else 0, lambda t : 1 if t > 5 else 0
    WhittleIdx = policy.Whittle([l1, l2], [mu1, mu2], [c1, c2])
    run_2Class_MG1_tests("MM1WhIdx", l1, l2, lib.exp(mu1), lib.exp(mu2), WhittleIdx)

    c1, c2 = lambda t : 4, lambda t : 2
    WhittleIdx = policy.Whittle([l1, l2], [mu1, mu2], [c1, c2])
    run_2Class_MG1_tests("MM1WhIdx", l1, l2, lib.exp(mu1), lib.exp(mu2), WhittleIdx)
