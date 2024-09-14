import math, random
import numpy as np

## moments from samples needed for derivations

def moments_from_samples(samples, NMOMENTS=4):
    moments = [np.mean([S**k for S in samples]) for k in range(NMOMENTS)]
    return moments

def moments_from_sample_gen(sample_generator, NSAMPLES=10**6, NMOMENTS=4):
    samples = [sample_generator() for _ in range(NSAMPLES)]
    moments = [np.mean([S**k for S in samples]) for k in range(NMOMENTS)]
    return moments

def Csq_from_moments(moments):
    return moments[2] / moments[1]**2 - 1

## common random variable generators

# random variable generators
def exp(mu):
    return lambda:random.expovariate(mu)

def det(mu):
    return lambda:1/mu 

def hyperexponential(mu, Csq):
    p = 0.5 * (1 + math.sqrt((Csq - 1)/ (Csq + 1)))
    mu1, mu2 = 2 * p * mu, 2 * (1-p) * mu
    def gen():
        if np.random.uniform() < p:
            return random.expovariate(mu1)
        else:
            return random.expovariate(mu2)
    return gen

def pareto(mu, k = 1965.5, p = 10**10, a = 2.9): # ES = 3000 c = 1/mu => c = 1/3000/mu
    return lambda : (1/3000/mu)*(k**(-a) - (1-(k/p)**a)/k**a * np.random.uniform()) ** (-1/a)
