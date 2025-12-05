import numpy as np

def curlfree_poly(x, l):
    n = x.shape[0]
    if l == 1:
        CP = np.zeros((3*n, 3))
        P = np.zeros((n, 3))
        P[:, 0:3] = x
        CP[:, 0] = np.tile(np.array([1, 0, 0]), n)
        CP[:, 1] = np.tile(np.array([0, 1, 0]), n)
        CP[:, 2] = np.tile(np.array([0, 0, 1]), n)
        return CP, P
    if l == 2:
        CP = np.zeros((3*n, 9))
        P = np.zeros((n, 9))
        P[:, 0:3] = x
        CP[:, 0] = np.tile(np.array([1, 0, 0]), n)
        CP[:, 1] = np.tile(np.array([0, 1, 0]), n)
        CP[:, 2] = np.tile(np.array([0, 0, 1]), n)
        P[:, 3:6] = 0.5 * x**2
        CP[:, 3] = np.concatenate([x[:, 0], np.zeros(n), np.zeros(n)])
        CP[:, 4] = np.concatenate([np.zeros(n), x[:, 1], np.zeros(n)])
        CP[:, 5] = np.concatenate([np.zeros(n), np.zeros(n), x[:, 2]])
        P[:, 6] = x[:, 1] * x[:, 2]
        CP[:, 6] = np.concatenate([np.zeros(n), x[:, 2], x[:, 1]])
        P[:, 7] = x[:, 0] * x[:, 2]
        CP[:, 7] = np.concatenate([x[:, 2], np.zeros(n), x[:, 0]])
        P[:, 8] = x[:, 0] * x[:, 1]
        CP[:, 8] = np.concatenate([x[:, 1], x[:, 0], np.zeros(n)])
        return CP, P
    raise ValueError('Degree not implemented')

