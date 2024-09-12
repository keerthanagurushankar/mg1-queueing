import math
import numpy as np

## random variable transformations

def excess(X):
    # input: list of 0..k moments of X: [1, EX, EX2 .. EXk]
    # output: list of 0..(k-1) moments of Xe
    if X[1] == 0:
        return X
    return [X[k]/k/X[1] for k in range(1, len(X))]

def two_case_rv(w1, w2, X1, X2):
    # returns (moments of) RV formed by casing on X1, X2 w weights w1, w2 resp.
    return [(w1 * X1[k] + w2 * X2[k])/(w1+w2) for k in range(min(len(X1), len(X2)))]

def rv_sum(X1, X2):
    # returns (moments of) RV X1 + X2
    EX = X1[1] + X2[1]
    EXsq = X1[2] + 2 * X1[1] * X2[1] + X2[2]
    return [1, EX, EXsq]

## common random variables

def moments_from_samples(samples, NMOMENTS=4):
    moments = [np.mean([S**k for S in samples]) for k in range(NMOMENTS)]
    return moments

def moments_from_sample_gen(sample_generator, NSAMPLES=10**6, NMOMENTS=4):
    samples = [sample_generator() for _ in range(NSAMPLES)]
    moments = [np.mean([S**k for S in samples]) for k in range(NMOMENTS)]
    return moments

def exp(mu, NMOMENTS=4):
    return [math.factorial(k) * (1/mu)**k for k in range(NMOMENTS)]

def det(D, NMOMENTS=4):
    return [1, D] + [0] * (NMOMENTS-2)

## MG1 Formulas

class BP:
    def __init__(self, W0, l, S):
        # BP with initial work W0, rate l of jobs S
        self.W0 = W0
        self.l = l
        self.S = S
        
        self.rho = l * S[1]
        self.W0e = excess(W0)
        self.Se = excess(S)
        
    def BP_length(self):
        # returns (moments of) length of busy period
        EBP = self.W0[1] / (1-self.rho)
        EBPe = (self.W0e[1] + self.rho/(1-self.rho)*self.Se[1]) / (1-self.rho)
        EBPsq = EBPe * 2 * EBP
        return [1, EBP, EBPsq]
    
    def W(self):
        # returns (moments of) time-avg work during busy period
        # W = W0e + (Ws as sum[Geo[1-rho]-1][Se])
        # requires 4 moments of S to return 3 moments of W
        EWS = self.rho / (1-self.rho) * self.Se[1]
        EWSsq = 2 * (EWS)**2 + self.rho/(1-self.rho) * self.Se[2]
        WS = [1, EWS, EWSsq]
        W = rv_sum(self.W0e, WS)
        return W

class MG1:
    def __init__(self, l, S):
        self.l = l
        self.S = S
        self.Se = excess(S)
        self.rho = l * S[1]
        assert self.rho < 1, "Load must be less than 1"        

        self.BP = BP(S, l, S)
        self.W = two_case_rv(1-self.rho, self.rho, det(0), self.BP.W())

    def T_FCFS(self):
        # return rv_sum(self.W, self.S)
        EW = self.rho / (1-self.rho) * self.Se[1]
        EWsq = 2 * EW**2 + self.rho / (1-self.rho) * self.Se[2]
        return rv_sum([1, EW, EWsq], self.S)
    
class TwoClassMG1:
    def __init__(self, l1, l2, S1, S2):
        # requires at least 3 moments of S1, S2
        self.l1 = l1
        self.l2 = l2
        self.S1 = S1
        self.S2 = S2
        self.S1e = excess(S1)
        self.S2e = excess(S2)
        
        self.rho1 = l1 * S1[1]
        self.rho2 = l2 * S2[1]
        self.rho = self.rho1 + self.rho2
        self.S = two_case_rv(l1, l2, S1, S2)
        self.Se = excess(self.S)
        assert self.rho < 1, "Load must be less than 1"

        EW = self.rho / (1-self.rho) * self.Se[1]
        EWsq = 2 * EW**2 + self.rho / (1-self.rho) * self.Se[2]
        self.W = [1, EW, EWsq]

    def T_FCFS(self):
        return rv_sum(self.W, self.S)

    def T_PPrio12(self):
        T1 = MG1(self.l1, self.S1).T_FCFS()

        # W0 = W + S2; T2 = length of BP[W0, l1, S1]
        W0 = rv_sum(self.W, self.S2)
        Class1_BP = BP(W0, self.l1, self.S1)
        T2 = Class1_BP.BP_length()

        T = two_case_rv(self.l1, self.l2, T1, T2)
        return T1, T2

    def T_NPPrio12(self):
        # AI0 = BP[S, l1, S1], AI1 = BP[S2, l1, S1], BP = BP[AI0, l2, AI1]
        AI0 = BP(self.S, self.l1, self.S1)
        AI1 = BP(self.S2, self.l1, self.S1)
        p0 = (1-self.rho) / (1-self.rho1)
        TQ1p = two_case_rv(p0, 1-p0, AI0.W(), AI1.W())
        TQ1 = two_case_rv(1-self.rho, self.rho, [1, 0, 0], TQ1p)
        T1 = rv_sum(TQ1, self.S1)

        # TQ2 = length of BP[W, l1, S1]
        TQ2 = BP(self.W, self.l1, self.S1).BP_length()
        T2 = rv_sum(TQ2, self.S2)

        # compare with textbook tag job answers
        assert math.isclose(T1[1], self.rho/(1-self.rho1) * self.Se[1]
                            + self.S1[1]), "Wrong NPPrio ET1 derivation"
        assert math.isclose(T2[1], self.W[1] / (1-self.rho1)
                            + self.S2[1]), "Wrong NPPrio ET2 derivation"
        return T1, T2

    def T_ASHybrid(self, p):
        Ta, Tb = self.T_PPrio12()
        T1 = two_case_rv(1-p, p, Ta, Tb)
        T2 = two_case_rv(p, 1-p, Ta, Tb)
        return T1, T2


## plot

import matplotlib.pyplot as plt

def plot_MM1():
    mu1, mu2 = 10, 5
    S1 = [math.factorial(k) * (1/mu1)**k for k in range(4)]
    S2 = [math.factorial(k) * (1/mu2)**k for k in range(4)]    
    rho, FCFS, PPrio12, PPrio21, ASH = [], [], [], [], []

    for l in np.linspace(0.3, 1/(1/mu1+1/mu2)-0.1, 20) :
        rho.append(l*(1/mu1+1/mu2))
        MM12 = TwoClassMG1(l, l, S1, S2)
        MM21 = TwoClassMG1(l, l, S2, S1)
        
        FCFS.append(MM12.T_FCFS()[1])
        PPrio12.append(two_case_rv(l, l, *MM12.T_PPrio12())[1])
        PPrio21.append(two_case_rv(l, l, *MM21.T_PPrio12())[1])
        ASH.append(two_case_rv(l, l, *MM12.T_ASHybrid(0.6))[1])

    print(PPrio12, ASH)

    plt.plot(rho, FCFS, label='FCFS', linestyle='-')
    plt.plot(rho, PPrio12, label='PPrio12')
    plt.plot(rho, PPrio21, label='PPrio21')    
    plt.plot(rho, ASH, label='ASH', linestyle='--')
    plt.xlabel("Load")
    plt.ylabel("ET")
    plt.legend()
    plt.show()

def plot_MH1():
    mu1, mu2 = 0.4, 0.2
    

if __name__ == "__main__":
    l1, l2 = 0.15, 0.1
    mu1, mu2 = 0.4, 0.2
    S1 = [math.factorial(k) * (1/mu1)**k for k in range(4)]
    S2 = [math.factorial(k) * (1/mu2)**k for k in range(4)]
    MM1 = TwoClassMG1(l1, l2, S1, S2)

    print("FCFS: ", MM1.T_FCFS())
    print("PPrio12: ", two_case_rv(l1, l2, *MM1.T_PPrio12()))
    print("ASHybrid(1): ", two_case_rv(l1, l2, *MM1.T_ASHybrid(1)))
    
    plot_MM1()
    
