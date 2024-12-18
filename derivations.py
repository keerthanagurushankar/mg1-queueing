import math, random
import numpy as np
import lib

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
    
def scale_rv(c, X):
    # returns (moments of) c X where c is a constant
    return [c**k * Xk for k, Xk in enumerate(X)]

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
        
        if type(S) is list:
            if len(S) < 5: # if S input is given as moments
                self.S = S 
            else: # if S input is given as samples list
                self.S_samples = S
                self.S = lib.moments_from_samples(S)
        else: # if S input is given as sample generator function
            self.S_gen = S
            self.S = lib.moments_from_sample_gen(S)
            
        self.Se = excess(self.S)
        self.rho = self.l * self.S[1]
        assert self.rho < 1, "Load must be less than 1"

        self.BP = BP(self.S, self.l, self.S)
        self.W = two_case_rv(1-self.rho, self.rho, [1, 0, 0], self.BP.W())


    def T_FCFS(self):
        # return rv_sum(self.W, self.S)
        EW = self.rho / (1-self.rho) * self.Se[1]
        EWsq = 2 * EW**2 + self.rho / (1-self.rho) * self.Se[2]
        return rv_sum([1, EW, EWsq], self.S)

    def T_PSJF(self):
        self.S_gen = lambda _ : 0        
        Sx = lambda x : [S for S in S_samples if S <= x]
        Resx = lambda x : x/(1-self.l*np.mean(Sx(x)))
        pass
    
    def T_SRPT(self):
        if self.S_gen is None:
            raise Exception("Need job size generator to compute T_SRPT")
        
        S_samples = [self.S_gen() for _ in range(4 * 10**3)]
        Sxbar_samples = lambda x: [S if  S < x else x for S in S_samples]
        ESxbar_sq = lambda x : np.mean([S**2 for S in Sxbar_samples(x)])

        rhox = lambda x: self.l * np.mean([S if S < x else 0 for S in S_samples])
        EWaitx = lambda x: self.l / 2 * ESxbar_sq(x) / (1 - rhox(x))**2

        eps = 0.05
        xs = np.arange(0, 1, eps)        
        #rhoxs = [rhox(x) for x in xs] # TODO: memoize rhoxs
        EResx = lambda x: np.sum([eps/(1-rhox(t)) for t in np.arange(0, x, eps)])
        ET = np.mean([EWaitx(S) + EResx(S) for S in S_samples])
        return [1, ET]

    def T(self, policy_name):
        if policy_name == "FCFS":
            return self.T_FCFS()
        elif policy_name == "SRPT":
            return self.T_SRPT()
    
class TwoClassMG1:
    def __init__(self, l1, l2, S1, S2):
        self.l1 = l1
        self.l2 = l2
        self.S1 = S1
        self.S2 = S2
        self.S1e = excess(S1)
        self.S2e = excess(S2)
        assert len(S1) >= 4 and len(S2) >= 4, "requires first 3 moments of S1, S2"        
        
        self.rho1 = l1 * S1[1]
        self.rho2 = l2 * S2[1]
        self.rho = self.rho1 + self.rho2
        self.S = two_case_rv(l1, l2, S1, S2)
        self.Se = excess(self.S)
        assert self.rho < 1, "Load must be less than 1"

        self.MG1 = MG1(l1 + l2, self.S)
        self.W = self.MG1.W
        self.W_pos = self.MG1.BP.W()

    def T_FCFS(self):
        T1 = rv_sum(self.W, self.S1)
        T2 = rv_sum(self.W, self.S2)
        return T1, T2

    def T_PPrio12(self):
        T1 = MG1(self.l1, self.S1).T_FCFS()

        # T2 = length of BP[W + S2, l1, S1]
        Class1_BP = BP(rv_sum(self.W, self.S2), self.l1, self.S1)
        T2 = Class1_BP.BP_length()

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

    def T_NPAccPrio(self, b1, b2):
        assert b1 >= b2, "Assume class 1 is higher priority"    
        
        # lA = (1-b2/b1)*l1
        # AI0 = BP[S, lA, S1], AI1 = BP[SuA, lA, S1], BP = BP[AI0, l2 + b2/b1*l1, AI1]
        l_acc = (1 - b2/b1) * self.l1
        AI0 = BP(self.S, l_acc, self.S1)
        S_unacc = two_case_rv(self.l1/b1, self.l2/b2, self.S1, self.S2)
        AI1 = BP(S_unacc, l_acc, self.S1)
        
        p0 = (1 - self.rho) / (1- l_acc * self.S1[1]) # pr[entering AI0 | entering BP]
        TQ1_acc = two_case_rv(p0, 1-p0, AI0.W(), AI1.W()) # TQ of acc job
        
        # TQ2 = length of BP[W, l_acc, S1]
        TQ2_pos = BP(self.W_pos, l_acc, self.S1).BP_length()
        TQ2 = two_case_rv(1-self.rho, self.rho, [1, 0, 0], TQ2_pos)
        
        TQ1_unacc_pos = scale_rv(b2/b1, TQ2_pos) # b1 * TQ1_unacc = b2 * TQ2 = V_unacc
        TQ1_pos = two_case_rv(1 - b2/b1, b2/b1, TQ1_acc, TQ1_unacc_pos)
        TQ1 = two_case_rv(1-self.rho, self.rho, [1, 0, 0], TQ1_pos)
        
        T1, T2 = rv_sum(TQ1, self.S1), rv_sum(TQ2, self.S2)
        return T1, T2 # T1 is wrong, T2 is right (matches simulation)

    def T_ASHybrid(self, p):
        Ta, Tb = self.T_PPrio12()
        T1 = two_case_rv(1-p, p, Ta, Tb)
        T2 = two_case_rv(p, 1-p, Ta, Tb)
        return T1, T2

    def T_Lookahead(self, alpha):
        return None
        pass

    def T_timeavg(self, T12):
        T1, T2 = T12
        return two_case_rv(self.l1, self.l2, T1, T2)

    def T_split_capacity(self, p):
        # 1-p fraction capacity to class 1 => S1 -> S1/(1-p)
        T1 = MG1(self.l1, scale_rv(1/(1-p), self.S1)).T_FCFS()
        T2 = MG1(self.l2, scale_rv(1/p, self.S2)).T_FCFS()
        return T1, T2
        

    def T(self, policy_name):
        if policy_name == "PPrio12":
            return self.T_PPrio12()
        elif policy_name == "NPPrio12":
            return self.T_NPPrio12()
        elif policy_name == "FCFS":
            return self.T_FCFS()
        elif policy_name[0] == "ASH":
            return self.T_ASHybrid(policy_name[2])
        elif policy_name[0] == "NPAccPrio":
            b1, b2 = policy_name[1:]
            return self.T_NPAccPrio(b1, b2)
        elif policy_name[0] == "Lookahead":
            alpha = policy_name[1]
            return self.T_Lookahead(alpha)
        elif policy_name[0] == "SplitCapacity":
            # p in (self.rho1, 1-self.rho2)
            p = (self.rho1 + 1 - self.rho2)/2
            print(self.rho1, 1-self.rho2)
            return self.T_split_capacity(p)
        else:
            return None
            
## plot

import matplotlib.pyplot as plt

def plot_ETsq(plot_title, S1_gen, S2_gen):
    S1, S2 = lib.moments_from_sample_gen(S1_gen), lib.moments_from_sample_gen(S2_gen)
    policy_names = ["FCFS", "PPrio12", ("SplitCapacity", 0.4)]
    T_bypi_byrho = []
    rhos = np.linspace(0.4, 0.6, 10)
    
    for rho in rhos:
        l = rho/(1/mu1 + 1/mu2)
        MG1 = TwoClassMG1(l, l, S1, S2)
        T_bypi = [MG1.T_timeavg(MG1.T(policy))[2] for policy in policy_names]
        T_bypi_byrho.append(T_bypi)

    T_byrho_bypi = np.array(T_bypi_byrho).T

    ls = ['-', '-.', '--', ':']
    for i, policy in enumerate(policy_names):
        plt.plot(rhos, T_byrho_bypi[i], label=policy, ls=ls[i%len(ls)])
    plt.xlabel("Load")
    plt.ylabel("ETsq")
    plt.title(plot_title)
    plt.legend()
    plt.savefig(plot_title+'.png')
    plt.show()

if __name__ == "__main__":
    mu1, mu2 = 40, 2
    
    plot_ETsq(f"{mu1, mu2}-Hyperexp(10)", lib.hyperexponential(mu1, 10), lib.hyperexponential(mu2, 10))
    plot_ETsq(f"{mu1, mu2}-Exponential", lambda:random.expovariate(mu1), lambda:random.expovariate(mu2))
    
