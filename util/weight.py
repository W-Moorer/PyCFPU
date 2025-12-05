import numpy as np

def weight(r, delta, k):
    r = r / delta
    phi = np.zeros_like(r)
    if k == 0:
        id1 = r <= (1/3)
        phi[id1] = 0.75 - 2.25 * r[id1]**2
        id2 = (r > 1/3) & (r <= 1)
        phi[id2] = 1.125 * (1 - r[id2])**2
        return phi
    if k == 1:
        id1 = r <= (1/3)
        phi[id1] = -4.5 / delta**2
        id2 = (r > 1/3) & (r <= 1)
        phi[id2] = (-2.25 * (1 - r[id2]) / delta**2) * (1.0 / r[id2])
        return phi
    raise ValueError('PU Weight function error')

