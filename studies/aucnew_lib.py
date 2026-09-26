import numpy as np
def auc(y, z):
    o = np.argsort(z, kind='mergesort'); r = np.empty(len(z)); r[o] = np.arange(1, len(z)+1)
    # ties
    zs = z[o]; i = 0
    while i < len(z):
        j = i
        while j+1 < len(z) and zs[j+1] == zs[i]: j += 1
        r[o[i:j+1]] = 0.5*(i+1+j+1); i = j+1
    n1 = y.sum(); n0 = len(y)-n1
    return (r[y == 1].sum() - n1*(n1+1)/2)/(n1*n0)
def logit_fit(Xf, y, lam=1e-3):
    mu, sd = Xf.mean(0), Xf.std(0)+1e-12; Z = np.c_[np.ones(len(Xf)), (Xf-mu)/sd]; w = np.zeros(Z.shape[1])
    for _ in range(50):
        p = 1/(1+np.exp(-Z@w)); g = Z.T@(p-y) + lam*w; Hs = (Z*(p*(1-p))[:, None]).T@Z + lam*np.eye(len(w))
        w -= np.linalg.solve(Hs, g)
    return lambda Xn: np.c_[np.ones(len(Xn)), (Xn-mu)/sd]@w
L = lambda v: -np.log10(np.maximum(v, 1e-16))
