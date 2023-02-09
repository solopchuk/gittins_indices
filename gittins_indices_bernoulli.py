import numpy as np

# Gittins index is a probaility of a fictionary (safe) arm that makes you indifferent to play a (risky) arm,
# It can be found by brute force trying every possible p(reward) of the safe arm in a list in [0,1] with some step size,
# and solving planning by comparing the safe and the risky arms. Code adapted from Gittins, 2011.

gamma = .9 # discount rate
T = 6 # time horizon, i.e. tree depth (alpha + beta), for good results choose 1000, but it depends on gamma
step = 0.0001 # step-size of p qrid
R = np.zeros((T-1,T-1)) # Initialize an array of intermediate values
Gittins = np.zeros((T-1,T-1)) # Initialize an array of final Gittins indices

# initialize starting points for backward induction (endpoints)
for alpha in range(1,T): # T = alpha + beta
    R[alpha-1,T-alpha-1] = alpha/T # initialize endpoints at E[beta(a, b)] = alpha / (alpha + beta)

# Main loop
for p in np.arange(step/2, 1, step): 

    safe = p/(1-gamma); # value of the safe arm

    for t in np.arange(T-1, 1, -1): # t = alpha + beta, going backwards

        for alpha in np.arange(1,t): # enumerate all alphas up to alpha + beta

            # value of the risky arm
            risky = alpha/t * (1 + gamma * R[alpha,t-alpha-1]) + (t-alpha)/t * (gamma * R[alpha-1,t-alpha])
            
            # safe increases faster than risky in p,
            # so for a particular (alpha, beta), whenever safe is at least as good as risky,
            # we record the p of the safe as the Gittins index
            if (Gittins[alpha-1,t-alpha-1] == 0) and (safe>=risky):
                Gittins[alpha-1,t-alpha-1] = p-step/2

            # update for the recursion, because at every step you'd have to choose between
            # playing the safe or the risky arm
            R[alpha-1,t-alpha-1] = np.max([safe,risky])

print(Gittins)
