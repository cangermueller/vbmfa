import numpy as np

def generate_nonzero(n):
    effects = np.random.uniform(1.0, 2.0, size=n)
    signs = np.random.binomial(1, 0.5, size=n)
    effects[signs == 1.0] *= -1.0
    return effects

def generate_factor_cluster(P, Q=0, mean0=None, N=10, scale=1.0):
    y = np.empty((N, P))
    P = int(P)
    Q = int(Q)
    l = generate_nonzero(int(P*Q)).reshape(P, Q)
    x = generate_nonzero(int(Q*N)).reshape(Q, N)
    if mean0 is None:
        mean0 = np.zeros(P)
    mean = np.transpose(l.dot(x)) + mean0
    cov = np.eye(P)*scale
    for n in range(N):
        y[n, :] = np.random.multivariate_normal(mean=mean[n, :], cov=cov, size=1)
    return [np.transpose(y), l, x]

def generate_clusters(P, Q=0, mean0=[None], N=[10], scale=1.0):
    num_clusters = len(mean0)
    if Q == 0:
        Q = np.empty(num_clusters)
        Q.fill(0)
    ys = np.empty((P, 0))
    for i in range(num_clusters):
        y, l, x = generate_factor_cluster(P, Q[i], mean0[i], N[i], scale)
        ys = np.hstack((ys, y))
    return ys

