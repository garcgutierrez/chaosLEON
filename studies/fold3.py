import numpy as np, numba as nb, math
from h3 import WW, Y1, Y0
from pci3 import proj3
from fold import wedge

@nb.njit(cache=True)
def step2(s, w, v, dt, n, om):
    """orbit + first (w) + second (v) variation, exact discrete variation of Yoshida leapfrog."""
    for _ in range(n):
        for c in (Y1, Y0, Y1):
            h = 0.5*c*dt
            for half in range(2):
                q0, q1, q2 = s[0], s[1], s[2]
                a = om[0] + 2*q1 + 2*q2; b = 2*q0
                d0, d1, d2 = w[0], w[1], w[2]
                v[3] -= h*(a*v[0] + b*v[1] + b*v[2] + 4*d0*(d1 + d2)); v[4] -= h*(b*v[0] + om[1]*v[1] + 2*d0*d0); v[5] -= h*(b*v[0] + om[2]*v[2] + 2*d0*d0)
                w[3] -= h*(a*d0 + b*d1 + b*d2); w[4] -= h*(b*d0 + om[1]*d1); w[5] -= h*(b*d0 + om[2]*d2)
                s[3] -= h*(om[0]*q0 + 2*q0*q1 + 2*q0*q2); s[4] -= h*(om[1]*q1 + q0*q0); s[5] -= h*(om[2]*q2 + q0*q0)
                if half == 0:
                    for i in range(3):
                        s[i] += 2*h*om[i]*s[3+i]; w[i] += 2*h*om[i]*w[3+i]; v[i] += 2*h*om[i]*v[3+i]

@nb.njit(cache=True)
def fold3_run(x0, u0, dt, tau, nbk, hidx, om):
    s = x0.copy(); w = proj3(s, u0.copy(), om); w /= np.sqrt(np.sum(w*w)); v = np.zeros(6)
    Lc = 0.0; supfi = -1e300; ofli2 = -1e300; hk = 0; out = np.full((hidx.size, 2), np.nan)
    for b in range(nbk):
        step2(s, w, v, dt, tau, om)
        pw = proj3(s, w, om); pv = proj3(s, v, om); npw = math.sqrt(np.sum(pw*pw))
        supfi = max(supfi, Lc + math.log(max(wedge(pw, pv), 1e-300)) - 2*math.log(npw))
        z = pw + 0.5*math.exp(min(Lc, 600.0))*pv; ofli2 = max(ofli2, Lc + math.log(math.sqrt(np.sum(z*z))))
        if hk < hidx.size and b == hidx[hk]:
            out[hk, 0] = supfi; out[hk, 1] = ofli2; hk += 1
        n = math.sqrt(np.sum(w*w)); Lc += math.log(n); w /= n; v /= n*n
    return out

@nb.njit(parallel=True, cache=True)
def fold3_all(X, U, dt, tau, nbk, hidx, om):
    out = np.empty((X.shape[0], hidx.size, 2))
    for i in nb.prange(X.shape[0]):
        out[i] = fold3_run(X[i], U[i], dt, tau, nbk, hidx, om)
    return out
