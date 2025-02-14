import math, random
import numpy as np

# import simulations
# import time_based_sims as simulations
import event_based_sims as simulations
import lib, policies as policy, derivations
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

def run_basic_tests(l, mu, l1, l2, mu1, mu2):
    print("**MG1 FCFS TESTS**")    
    run_MG1_tests("MM1", l, lib.exp(mu))
    run_MG1_tests("MD1", l, lib.det(mu))
    run_MG1_tests("MH1", l, lib.hyperexponential(mu,Csq=5))
    run_MG1_tests("MPar1",l,lib.pareto(mu))
    run_2Class_MG1_tests("2cMM1", l1, l2, lib.exp(mu1), lib.exp(mu2), policy.FCFS)

    print("**2 CLASS NPPRIO TESTS**")
    run_2Class_MG1_tests("2cMM1a", l1, l2, lib.exp(mu1), lib.exp(mu2), policy.NPPrio12)
    run_2Class_MG1_tests("2cMM1b", l2, l1, lib.exp(mu2), lib.exp(mu1), policy.NPPrio12)
    run_2Class_MG1_tests("2cMD1a", l1, l2, lib.det(mu1), lib.det(mu2), policy.NPPrio12)
    run_2Class_MG1_tests("2cMD1b", l2, l1, lib.det(mu2), lib.det(mu1), policy.NPPrio12)
    run_2Class_MG1_tests("2cMH1a", l1, l2, lib.hyperexponential(mu1, Csq=5),
                        lib.hyperexponential(mu2, Csq=10), policy.NPPrio12)


    print("**2 CLASS PPRIO TESTS**")
    run_2Class_MG1_tests("2cMM1c", l1, l2, lib.exp(mu1), lib.exp(mu2), policy.PPrio12)
    run_2Class_MG1_tests("2cMM1d", l2, l1, lib.exp(mu2), lib.exp(mu1), policy.PPrio12)
    run_2Class_MG1_tests("2cMD1c", l1, l2, lib.det(mu1), lib.det(mu2), policy.PPrio12)
    run_2Class_MG1_tests("2cMH1c", l1, l2, lib.hyperexponential(mu1, Csq=5),
                        lib.hyperexponential(mu2, Csq=5), policy.PPrio12)
    run_2Class_MG1_tests("2cMH1d", l1, l2, lib.hyperexponential(mu1, Csq=10),
                        lib.hyperexponential(mu2, Csq=10), policy.PPrio12)

    print("**SRPT TESTS**")
    l, mu = 7, 10
    # run_MG1_tests("SRPT MM1", l, lib.exp(mu), policy.SRPT)    
    # run_MG1_tests("SRPT MD1", l, lib.det(mu), policy.SRPT)
    # run_MG1_tests("SRPT MH1 Csq10", l, lib.hyperexponential(mu, Csq=10), policy.SRPT)
    # run_MG1_tests("SRPT MH1 Csq50", l, lib.hyperexponential(mu, Csq=50), policy.SRPT)    
    # run_MG1_tests("SRPT MH1 Csq100", l, lib.hyperexponential(mu, Csq=100), policy.SRPT)
    # run_MG1_tests("SRPT MPar1", l, lib.pareto(mu), policy.SRPT)

def run_linear_tests(l1, l2, mu1, mu2):
    print("**2 CLASS NP-ACC-PRIO TESTS**")
    # NPAccPrio = policy.AccPrio(b1 = 3, b2 = 2, is_preemptive = False) 
    # run_2Class_MG1_tests("2cMM1NPacc", l1, l2, lib.exp(mu1), lib.exp(mu2), NPAccPrio)
    # run_2Class_MG1_tests("2cMD1NPacc", l1, l2, lib.det(mu1), lib.det(mu2), NPAccPrio)
    # run_2Class_MG1_tests("2cMH1NPacc", l1, l2, lib.hyperexponential(mu1, Csq=5),
    #                     lib.hyperexponential(mu2, Csq=5), NPAccPrio)
    b1, b2 = 1, 1 

    print("**2 CLASS P-ACC-PRIO TESTS**")
    PAccPrio = policy.AccPrio(b1 = 3, b2 = 2, is_preemptive = True)
    run_2Class_MG1_tests("2cMM1Pacc", l1, l2, lib.exp(mu1), lib.exp(mu2), PAccPrio)
    run_2Class_MG1_tests("2cMD1Pacc", l1, l2, lib.det(mu1), lib.det(mu2), PAccPrio)
    run_2Class_MG1_tests("2cMH1Pacc", l1, l2, lib.hyperexponential(mu1, Csq=5),
                         lib.hyperexponential(mu2, Csq=5), PAccPrio)  
    PAccPrio = policy.AccPrio(b1 = 100, b2 = 1, is_preemptive = True)
    run_2Class_MG1_tests("2cMM1PaccB", l1, l2, lib.exp(mu1), lib.exp(mu2), PAccPrio)

    print("**LOOKAHEAD TESTS**")
    run_2Class_MG1_tests("MM1Look0", l1, l2, lib.exp(mu1), lib.exp(mu2), policy.Lookahead(0))
    run_2Class_MG1_tests("MM1Look10",l1, l2, lib.exp(mu1), lib.exp(mu2), policy.Lookahead(10))
    c1, d1, c2 = 10, 10, 1 #mu1 c1 exp(-(mu1 - l1) * (d1 - alpha_star)) = mu2 c2
    alpha_star = d1 - np.log((mu1*c1)/(mu2*c2)) / (mu1 - l1)
    print(f"alpha_star {alpha_star}")
    run_2Class_MG1_tests("MM1Look*",l1, l2, lib.exp(mu1), lib.exp(mu2), policy.Lookahead(alpha_star))

    print("**2 CLASS LINEAR WHITTLE TESTS**")
    # Define Linear Whittle Policy with different parameters
    LinearWhittle1 = policy.LinearWhittle(arrival_rates=[l1, l2], service_rates=[mu1, mu2], cost_rates=[(3, 2), (2, 1)])
    run_2Class_MG1_tests("2cMM1LWhittle", l1, l2, lib.exp(mu1), lib.exp(mu2), LinearWhittle1)
    run_2Class_MG1_tests("2cMD1LWhittle", l1, l2, lib.det(mu1), lib.det(mu2), LinearWhittle1)
    run_2Class_MG1_tests("2cMH1LWhittle", l1, l2, lib.hyperexponential(mu1, Csq=5),
                         lib.hyperexponential(mu2, Csq=5), LinearWhittle1)

    # Test with more extreme cost ratios
    LinearWhittle2 = policy.LinearWhittle(arrival_rates=[l1, l2], service_rates=[mu1, mu2], cost_rates=[(100, 1), (5, 1)])
    run_2Class_MG1_tests("2cMM1LWhittleB", l1, l2, lib.exp(mu1), lib.exp(mu2), LinearWhittle2)
    run_2Class_MG1_tests("2cMD1LWhittleB", l1, l2, lib.det(mu1), lib.det(mu2), LinearWhittle2)
    run_2Class_MG1_tests("2cMH1LWhittleB", l1, l2, lib.hyperexponential(mu1, Csq=5),
                         lib.hyperexponential(mu2, Csq=5), LinearWhittle2)

def run_quadratic_tests(l1, l2, mu1, mu2):
    print("**2 CLASS P-QUAD-ACC-PRIO TESTS**")
    PQAP1 = policy.QuadraticAccPrio([1, 2], [0, 0], [0, 0])
    PQAP2 = policy.QuadraticAccPrio([1, 0], [0, 30], [0, 0])
    run_2Class_MG1_tests("PQAP1", l1, l2, lib.exp(mu1), lib.exp(mu2), PQAP1)
    run_2Class_MG1_tests("PQAP2", l1, l2, lib.exp(mu1), lib.exp(mu2), PQAP2)
    
    
    print("**2 CLASS QUADRATIC POLICY M/M/1 TESTS**")
    load_conditions = [
        (0.2 * mu1, 0.2 * mu2),  # Low Load
        (0.3 * mu1, 0.3 * mu2),  # Medium Load
        (0.4 * mu1, 0.4 * mu2)   # High Load
    ]

    # Define quadratic cost coefficients (a, b, c) per job class
    cost_rates = [(0.5, 1.0, 2.0), (0.4, 0.8, 1.5)]
    cost_rates = [(1, 2, 1), (0.5, 1, 2)]

    for idx, (l1, l2) in enumerate(load_conditions):
        # Test Quadratic Whittle Policy
        QuadWhittle = policy.QuadraticWhittle(arrival_rates=[l1, l2], service_rates=[mu1, mu2], cost_rates=cost_rates)
        test_name_whittle = f"2cMM1QWhittle_Load{idx+1}"
        run_2Class_MG1_tests(test_name_whittle, l1, l2, lib.exp(mu1), lib.exp(mu2), QuadWhittle)

        # Test Quadratic Aalto Policy
        QuadAalto = policy.QuadraticAalto(arrival_rates=[l1, l2], service_rates=[mu1, mu2], cost_rates=cost_rates)
        test_name_aalto = f"2cMM1QAalto_Load{idx+1}"
        run_2Class_MG1_tests(test_name_aalto, l1, l2, lib.exp(mu1), lib.exp(mu2), QuadAalto)

def run_age_based_tests(l1, l2, mu1, mu2):
    AgeBasedPQAP2 = policy.AgeBasedPrio(lambda r, s, t, k: t**2 if k==0 else t+30)
    run_2Class_MG1_tests("ABPQAP2", l1, l2, lib.exp(mu1), lib.exp(mu2), AgeBasedPQAP2)

    print("**WHITTLE INDEX TESTS**")
    # Fixed holding costs: Whittle is PPrio12
    c1, c2 = lambda t : 10, lambda t : 1
    WhittleIdx = policy.Whittle([l1, l2], [mu1, mu2], [c1, c2])
    run_2Class_MG1_tests("MM1WhIdxA", l1, l2, lib.exp(mu1), lib.exp(mu2), WhittleIdx)
    # Deadline c1 + fixed c2: Whittle is a lookahead policy
    c1, c2 = lambda t : 10 if t > 50/3 else 0, lambda t : 1
    WhittleIdx = policy.Whittle([l1, l2], [mu1, mu2], [c1, c2])
    run_2Class_MG1_tests("MM1WhIdxB", l1, l2, lib.exp(mu1), lib.exp(mu2), WhittleIdx)
    # Deadlines with weighted penalties
    c1, c2 = lambda t : 10 if t > 50/3 else 0, lambda t : 1 if t > 5 else 0
    WhittleIdx = policy.Whittle([l1, l2], [mu1, mu2], [c1, c2])
    run_2Class_MG1_tests("MM1WhIdxC", l1, l2, lib.exp(mu1), lib.exp(mu2), WhittleIdx)
    # Growing penalty for class 1 but cost rates are still eventually constant
    c1, c2 = lambda t : 0.6 * t if t < 50/3 else 1, lambda t : 1 if t > 5 else 0
    WhittleIdx = policy.Whittle([l1, l2], [mu1, mu2], [c1, c2])
    run_2Class_MG1_tests("MM1WhIdxD", l1, l2, lib.exp(mu1), lib.exp(mu2), WhittleIdx)

def run_age_based_tests2():
    print("**2 CLASS AGE-BASED POLICY M/M/1 TESTS**")

    mu1, mu2, c1, c2, d1, d2 = 2, 2, 3, 3, 7, 7    
    age_values = np.linspace(0, max(d1, d2)*1.1, 20)

    
    # Define different load conditions (Low, Medium, High)
    load_conditions = [
        (0.2 * mu1, 0.2 * mu2),  # Low Load
        (0.3 * mu1, 0.3 * mu2),  # Medium Load
        (0.4 * mu1, 0.4 * mu2)   # High Load
    ]

    # Define holding cost functions (quadratic costs)
    holding_cost_rates = [
        lambda t: t**2 + 2*t + 1,  # Quadratic cost for class 1
        lambda t: 0.5*t**2 + t + 2  # Quadratic cost for class 2
    ]

    holding_cost_rates = [
        lambda t : c1 if t > d1 else 0,
        lambda t : c2 if t > d2 else 0
    ]

    for idx, (l1, l2) in enumerate(load_conditions[-1:]):
        assert l1/mu1 + l2/mu2 < 1, "Load must be less than 1"
        
        # Test generalized cμ Policy
        GenCMU = policy.generalized_cmu(service_rates=[mu1, mu2], holding_cost_rates=holding_cost_rates)
        test_name_cmu = f"2cMM1GenCMU_Load{idx+1}"
        #run_2Class_MG1_tests(test_name_cmu, l1, l2, lib.exp(mu1), lib.exp(mu2), GenCMU)

        # Test Whittle Policy
        WhittlePolicy = policy.Whittle(arrival_rates=[l1, l2], service_rates=[mu1, mu2],
                                       holding_cost_rates=holding_cost_rates, age_values = age_values)
        test_name_whittle = f"2cMM1Whittle_Load{idx+1}"
        run_2Class_MG1_tests(test_name_whittle, l1, l2, lib.exp(mu1), lib.exp(mu2), WhittlePolicy)

        # Test Aalto Policy
        AaltoPolicy = policy.Aalto(arrival_rates=[l1, l2], service_rates=[mu1, mu2], holding_cost_rates=holding_cost_rates)
        test_name_aalto = f"2cMM1Aalto_Load{idx+1}"
        #run_2Class_MG1_tests(test_name_aalto, l1, l2, lib.exp(mu1), lib.exp(mu2), AaltoPolicy)

def run_gittins_tests():
    print("*** GITTINS EASY TESTS ***")
    # # Fixed identical holding costs test of Gittins and FCFS
    # l1, mu1 = 3/8, 3
    # c1, C1 = lambda t : 5, lambda t : 5*t
    # l2, mu2, c2, C2 = l1, mu1, c1, C1
    
    # GittinsIdx = policy.iterativeGittins([l1, l2], [mu1, mu2], [c1, c2], [C1, C2], 10, gttns_fn=gttns_fn)
    # run_2Class_MG1_tests("Gittins", l1, l2, lib.exp(mu1), lib.exp(mu2), GittinsIdx)
    # run_2Class_MG1_tests("FCFS", l1, l2, lib.exp(mu1), lib.exp(mu2), policy.FCFS)

    # # One deadline identical classes test of Gittins and FCFS
    # l1, mu1 = 3/8, 3
    # c1, C1 = lambda t : 5 if t > 10 else 0, lambda t : 0 if t < 10 else 5*(t-10)
    # l2, mu2, c2, C2 = l1, mu1, c1, C1
    
    # GittinsIdx = policy.iterativeGittins([l1, l2], [mu1, mu2], [c1, c2], [C1, C2], 10, gttns_fn=gttns_fn)
    # run_2Class_MG1_tests("Gittins", l1, l2, lib.exp(mu1), lib.exp(mu2), GittinsIdx)
    # run_2Class_MG1_tests("FCFS", l1, l2, lib.exp(mu1), lib.exp(mu2), policy.FCFS)

    # One deadline test of Gittins and Whittle
    l1, l2, mu1, mu2 = 3/8, 3/8, 3, 1
    c1, C1 = lambda t : 5 if t > 10 else 0, lambda t : 0 if t < 10 else 5*(t-10)
    c2, C2 = lambda t : 1 if t > 5 else 0, lambda t : 0 if t < 5 else t-5

    cmu = policy.generalized_cmu([mu1, mu2], [c1, c2])
    GittinsIdx = policy.iterativeGittins([l1, l2], [mu1, mu2], [c1, c2], [C1, C2], 25, alpha=0.1, initialPolicy=cmu, gttns_fn=gttns_fn)
    policy.MG1_ECost_tests("Gittins", [l1, l2], [mu1, mu2], [C1, C2], GittinsIdx)
    WhittleIdx = policy.Whittle([l1, l2], [mu1, mu2], [c1, c2])
    policy.MG1_ECost_tests("Whittle", [l1, l2], [mu1, mu2], [C1, C2], WhittleIdx)

def run_gittins_tests2():
    print("*** GITTINS HARD TEST ***")    
    # Gittins vs. Whittle on weird functions
    l1, l2, mu1, mu2 = 1.8, 0.2, 3, 3
    # dead1 = lambda t : 5 if t > 8 else 0, lambda t : 0 if t < 10 else 5*(t-8)
    # dead2 = lambda t : 5 if t > 10 else (1 if t > 5 else 0), lambda t : 5*(t-10)+5 if t > 10 else (t-5 if t > 5 else 0)
    # dead3 = lambda t : 3 if t > 7 else (2 if t > 3 else 0), lambda t : 3*(t-7)+8 if t > 7 else (2*(t-3) if t > 3 else 0)
    # quad1 = lambda t : t*t+2*t+1, lambda t : t*t*t/3+t*t+t
    # quad2 = lambda t : t*t+4*t+4, lambda t : t*t*t/3+2*t*t+4*t
    # osc = lambda t : math.exp(t) + math.sin(t), lambda t : math.exp(t) - math.cos(t)
    def mono(x): return lambda t : (x+1)*(t**x), lambda t : t**(x+1)

    c1, C1 = mono(10)
    c2, C2 = mono(10)

    GittinsIdx = policy.iterativeGittins([l1, l2], [mu1, mu2], [c1, c2], [C1, C2], 10)
    policy.MG1_ECost_tests("Gittins", [l1, l2], [mu1, mu2], [C1, C2], GittinsIdx)
    WhittleIdx = policy.Whittle([l1, l2], [mu1, mu2], [c1, c2])
    policy.MG1_ECost_tests("Whittle", [l1, l2], [mu1, mu2], [C1, C2], WhittleIdx)

def run_age_comp_test(l1, l2, mu1, mu2, a_vals, b_vals, c_vals):
    linearPolicy = policy.LinearAccPrio(b_vals, a_vals)
    run_2Class_MG1_tests("Linear Policy", l1, l2, lib.exp(mu1), lib.exp(mu2), linearPolicy)
    V = lambda r, s, t, k: a_vals[k-1]*t+b_vals[k-1]
    ageLinearPolicy = policy.AgeBasedPrio(V, num_classes=len(a_vals))
    run_2Class_MG1_tests("Age Linear Policy", l1, l2, lib.exp(mu1), lib.exp(mu2), ageLinearPolicy)
    
    quadPolicy = policy.QuadraticAccPrio(a_vals, b_vals, c_vals)
    run_2Class_MG1_tests("Quad Policy", l1, l2, lib.exp(mu1), lib.exp(mu2), quadPolicy)
    V = lambda r, s, t, k: a_vals[k-1]*t*t+b_vals[k-1]*t+c_vals[k-1]
    ageQuadPolicy = policy.AgeBasedPrio(V, num_classes=len(a_vals))
    run_2Class_MG1_tests("Age Quad Policy", l1, l2, lib.exp(mu1), lib.exp(mu2), ageQuadPolicy)

if __name__ == "__main__":
    l, mu = .4, 2
    l1, l2, mu1, mu2 = 3/8, 3/8, 3, 1    # 0.15, 0.45; 0.65625
    
    # run_basic_tests(l, mu, l1, l2, mu1, mu2)
    # run_linear_tests(l1, l2, mu1, mu2)
    # run_quadratic_tests(l1, l2, mu1, mu2)
    # run_age_based_tests(l1, l2, mu1, mu2)
    # run_age_based_tests2()
    # run_gittins_tests()
    run_gittins_tests(policy.inst_gittins)
    # run_gittins_tests2()
    # run_age_comp_test(l1, l2, mu1, mu2, [1, 2], [0, 0], [0, 0])
