import numpy as np

def gcv_cost_function(lam, z, d, n):
    lam = np.exp(-lam)
    temp = (n * lam) / (d**2 + n * lam)
    score = n * np.sum((temp * z)**2) / (np.sum(temp)**2)
    return score

