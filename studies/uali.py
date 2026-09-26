"""
Exploration: SALI vs a Gauss-Hermite (non-intrusive PCE) alignment index
for the Henon-Heiles system, E = 1/8.

UALI_sigma: first-order Hermite PC coefficients of the flow over a block,
a_i = E[ phi_tau(x_c + sigma * U xi) * xi_i ]  (pseudo-spectral projection)
    = sigma * E[ D phi_tau(x_c + sigma U xi) ] U e_i   (Stein's lemma)
i.e. a Gaussian-averaged tangent map.  SALI is evaluated on the normalised
columns a_i/|a_i|; the cloud is then re-seeded around the reference orbit
with the new (normalised) directions (Benettin-type renormalisation).
sigma -> 0 recovers SALI exactly.
"""
import numpy as np, numba as nb

W1 = 1.0/(2.0-2.0**(1.0/3.0)); W0 = -2.0**(1.0/3.0)/(2.0-2.0**(1.0/3.0))
YC = np.array([W1, W0, W1])

@nb.njit(cache=True, fastmath=False)
def lf(q0, q1, p0, p1, h):
    p0 -= 0.5*h*(q0 + 2*q0*q1); p1 -= 0.5*h*(q1 + q0*q0 - q1*q1)
    q0 += h*p0; q1 += h*p1
    p0 -= 0.5*h*(q0 + 2*q0*q1); p1 -= 0.5*h*(q1 + q0*q0 - q1*q1)
    return q0, q1, p0, p1

@nb.njit(cache=True)
def y4(s, dt, nsteps):
    q0, q1, p0, p1 = s[0], s[1], s[2], s[3]
    for _ in range(nsteps):
        for c in (W1, W0, W1):
            q0, q1, p0, p1 = lf(q0, q1, p0, p1, c*dt)
    s[0] = q0; s[1] = q1; s[2] = p0; s[3] = p1

@nb.njit(cache=True)
def y4_tan(s, W, dt, nsteps):
    """orbit + k tangent vectors (columns of W, shape 4xk), Yoshida-4 leapfrog."""
    q0, q1, p0, p1 = s[0], s[1], s[2], s[3]
    k = W.shape[1]
    for _ in range(nsteps):
        for c in (W1, W0, W1):
            h = c*dt
            a = 1 + 2*q1; b = 2*q0; d = 1 - 2*q1
            for j in range(k):
                W[2, j] -= 0.5*h*(a*W[0, j] + b*W[1, j]); W[3, j] -= 0.5*h*(b*W[0, j] + d*W[1, j])
            p0 -= 0.5*h*(q0 + 2*q0*q1); p1 -= 0.5*h*(q1 + q0*q0 - q1*q1)
            q0 += h*p0; q1 += h*p1
            for j in range(k):
                W[0, j] += h*W[2, j]; W[1, j] += h*W[3, j]
            a = 1 + 2*q1; b = 2*q0; d = 1 - 2*q1
            for j in range(k):
                W[2, j] -= 0.5*h*(a*W[0, j] + b*W[1, j]); W[3, j] -= 0.5*h*(b*W[0, j] + d*W[1, j])
            p0 -= 0.5*h*(q0 + 2*q0*q1); p1 -= 0.5*h*(q1 + q0*q0 - q1*q1)
    s[0] = q0; s[1] = q1; s[2] = p0; s[3] = p1

@nb.njit(cache=True)
def sali2(W):
    e1 = W[:, 0]/np.sqrt(np.sum(W[:, 0]**2)); e2 = W[:, 1]/np.sqrt(np.sum(W[:, 1]**2))
    sp = np.sqrt(np.sum((e1+e2)**2)); sm = np.sqrt(np.sum((e1-e2)**2))
    W[:, 0] = e1; W[:, 1] = e2
    return min(sp, sm)

@nb.njit(cache=True)
def run_sali(x0, dt, tau_steps, nblocks, W0_):
    s = x0.copy(); W = W0_.copy()
    out = np.empty(nblocks)
    for b in range(nblocks):
        y4_tan(s, W, dt, tau_steps)
        out[b] = sali2(W)
    return out

@nb.njit(cache=True)
def run_uali(x0, dt, tau_steps, nblocks, U0, sigma, nodes, wts):
    """nodes: (K,2) standard-normal GH nodes, wts: (K,) weights (sum 1).
    Returns SALI of the normalised first-order PC coefficients per block."""
    s = x0.copy(); U = U0.copy()
    K = nodes.shape[0]
    X = np.empty((K, 4))
    out = np.empty(nblocks)
    for b in range(nblocks):
        for k in range(K):
            for i in range(4):
                X[k, i] = s[i] + sigma*(U[i, 0]*nodes[k, 0] + U[i, 1]*nodes[k, 1])
        for k in range(K):
            y4(X[k], dt, tau_steps)
        y4(s, dt, tau_steps)             # reference orbit (pinned mean)
        A = np.zeros((4, 2))
        for k in range(K):
            for j in range(2):
                for i in range(4):
                    A[i, j] += wts[k]*X[k, i]*nodes[k, j]
        out[b] = sali2(A)                 # normalises columns in place
        U = A
    return out

def gh_nodes(n):
    x, w = np.polynomial.hermite_e.hermegauss(n)   # probabilists' Hermite
    w = w/np.sum(w)
    X, Y = np.meshgrid(x, x, indexing='ij'); WX, WY = np.meshgrid(w, w, indexing='ij')
    return np.stack([X.ravel(), Y.ravel()], 1), (WX*WY).ravel()

def px_from(y, py, E=0.125):
    v = 2*E - py**2 - y**2 + 2*y**3/3
    return np.sqrt(v) if v >= 0 else np.nan

@nb.njit(cache=True)
def gradH(s):
    x, y, px, py = s[0], s[1], s[2], s[3]
    return np.array([x + 2*x*y, y + x*x - y*y, px, py])

@nb.njit(cache=True)
def Hval(s):
    x, y, px, py = s[0], s[1], s[2], s[3]
    return 0.5*(px*px+py*py) + 0.5*(x*x+y*y) + x*x*y - y**3/3.0

@nb.njit(cache=True)
def run_uali_iso(x0, dt, tau_steps, nblocks, U0, sigma, nodes, wts, iso):
    """iso=1: seed directions projected on the energy-surface tangent space and
    every node's momentum rescaled onto H = H(x_c) (isoenergetic cloud)."""
    s = x0.copy(); U = U0.copy(); E = Hval(s)
    K = nodes.shape[0]
    X = np.empty((K, 4)); out = np.empty(nblocks)
    for b in range(nblocks):
        if iso == 1:
            g = gradH(s); g = g/np.sqrt(np.sum(g*g))
            f = np.array([s[2], s[3], -(s[0]+2*s[0]*s[1]), -(s[1]+s[0]**2-s[1]**2)])
            f = f - np.dot(f, g)*g; f = f/np.sqrt(np.sum(f*f))
            for j in range(2):
                u = U[:, j].copy()
                u = u - np.dot(u, g)*g
                u = u/np.sqrt(np.sum(u*u))
                U[:, j] = u
        for k in range(K):
            for i in range(4):
                X[k, i] = s[i] + sigma*(U[i, 0]*nodes[k, 0] + U[i, 1]*nodes[k, 1])
            if iso == 1:
                x, y = X[k, 0], X[k, 1]
                V = 0.5*(x*x+y*y) + x*x*y - y**3/3.0
                p2 = X[k, 2]**2 + X[k, 3]**2
                if 2*(E-V) > 0 and p2 > 0:
                    r = np.sqrt(2*(E-V)/p2); X[k, 2] *= r; X[k, 3] *= r
        for k in range(K):
            y4(X[k], dt, tau_steps)
        y4(s, dt, tau_steps)
        A = np.zeros((4, 2))
        for k in range(K):
            for j in range(2):
                for i in range(4):
                    A[i, j] += wts[k]*X[k, i]*nodes[k, j]
        out[b] = sali2(A)
        U = A
    return out
