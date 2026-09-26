# -*- coding: utf-8 -*-
"""
uali_henon_heiles.py  --  SALI vs covariance aspect ratio vs a finite-size,
non-intrusive polynomial-chaos alignment index (UALI_sigma) for Henon-Heiles, E=1/8.

Three quantities, all with a 4th-order Yoshida (symplectic) integrator, dt = 0.01,
renormalisation every tau = 1 time unit:

1) SALI  (variational equations, two deviation vectors e_x, e_px).

2) AR raw, fixed: aspect ratio sigma_2/sigma_1 of the square-root covariance
   L(t) = Phi(t) L0 (these are exactly the degree-1 PC coefficients for small
   sigma).  SVD of L instead of eig(C) keeps the full 1e-16 dynamic range, and the
   symplectic tangent map keeps C = L L^T positive semidefinite.  (The explicit
   update C <- C + dt (J C + C J^T) drops the dt^2 J C J^T term, C becomes
   indefinite and eig(C)_2 <= 0 gives the spurious AR = 0 at t ~ 36.)

   Identity: if the columns of L are normalised, AR_hat = tan(theta/2) and
   SALI = 2 sin(theta/2)  =>  AR_hat = SALI / sqrt(4 - SALI^2)   (exact).

3) UALI_sigma  (proposal): first-order Hermite PC coefficients of the block flow,
   obtained NON-INTRUSIVELY by Gauss-Hermite pseudo-spectral projection,
       a_i = E[ phi_tau(x_c + sigma U xi) xi_i ] = sigma E[ D phi_tau ] U e_i   (Stein)
   i.e. a Gaussian-averaged tangent map.  SALI of the normalised a_i, then the
   cloud is re-seeded around the reference orbit along the new directions
   (Benettin-style).  Isoenergetic version: directions projected on the energy
   tangent space and node momenta rescaled onto H = E.
   sigma -> 0 reproduces SALI; finite sigma asks "is the orbit chaotic at
   uncertainty level sigma?"  No variational equations are needed.
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


# ---------------------------------------------------------------------------
# grid tools (from grid.py)
# ---------------------------------------------------------------------------
@nb.njit(cache=True)
def ar_raw(x0, dt, tau_steps, nblocks):
    """Fixed version of the covariance aspect ratio: square-root (PC-coefficient)
    propagation with the symplectic tangent map, SVD instead of eig(C),
    trace renormalisation only (no per-column normalisation)."""
    s = x0.copy(); L = np.zeros((4, 2)); L[0, 0] = 1.0; L[2, 1] = 1.0
    ar = 1.0
    for b in range(nblocks):
        y4_tan(s, L, dt, tau_steps)
        sv = np.linalg.svd(L)[1]
        ar = sv[1]/sv[0]
        L = L/np.sqrt(np.sum(L*L))
    return ar

@nb.njit(cache=True)
def sali_T(x0, dt, tau_steps, nblocks):
    s = x0.copy(); W = np.zeros((4, 2)); W[0, 0] = 1.0; W[2, 1] = 1.0
    v = 1.0
    for b in range(nblocks):
        y4_tan(s, W, dt, tau_steps); v = sali2(W)
    return v

@nb.njit(parallel=True, cache=True)
def scan(Y, PY, PX, dt, tau, nbk, sigmas, nodes, wts):
    n = Y.size
    out = np.full((n, 2 + sigmas.size), np.nan)
    W0 = np.zeros((4, 2)); W0[0, 0] = 1.0; W0[2, 1] = 1.0
    for i in nb.prange(n):
        if np.isnan(PX[i]):
            continue
        x0 = np.array([0.0, Y[i], PX[i], PY[i]])
        out[i, 0] = sali_T(x0, dt, tau, nbk)
        out[i, 1] = ar_raw(x0, dt, tau, nbk)
        for j in range(sigmas.size):
            u = run_uali_iso(x0, dt, tau, nbk, W0.copy(), sigmas[j], nodes, wts, 1)
            out[i, 2 + j] = u[-1]
    return out

# ---------------------------------------------------------------------------
# CSALI: compensated section alignment index  (same cost as SALI)
# ---------------------------------------------------------------------------
# Remove the grad-H and flow components of both deviation vectors at every
# renormalisation.  The vectors then live in the 2D "section" plane (energy shell
# modulo the flow), i.e. the linearised Poincare map.  There the exponents are
# +lambda and -lambda instead of lambda and 0, so for chaotic orbits
#     SALI_perp ~ exp(-2 lambda t)      (SALI in 4D ~ exp(-lambda t)),
# while for regular orbits the twist shears the section ellipse and
#     SALI_perp ~ t^(-1..-2)            (cf. SALI of 2D maps ~ n^-2, Skokos & Manos 2016)
# CSALI(t) = t * SALI_perp(t) stays O(1)-O(10) for regular orbits.
# Benchmark (HH, E=1/8, 80x80): same accuracy as SALI with ~1.4x less CPU and
# ~1.5x less integration time (threshold 1e-3).
@nb.njit(cache=True)
def proj_perp(s, W):
    g = gradH(s)
    f = np.array([s[2], s[3], -(s[0]+2*s[0]*s[1]), -(s[1]+s[0]**2-s[1]**2)])
    g = g/np.sqrt(np.sum(g*g)); f = f - np.dot(f, g)*g; f = f/np.sqrt(np.sum(f*f))
    for j in range(W.shape[1]):
        w = W[:, j].copy()
        w = w - np.dot(w, g)*g; w = w - np.dot(w, f)*f
        W[:, j] = w/np.sqrt(np.sum(w*w))

@nb.njit(cache=True)
def run_csali(x0, dt, tau_steps, nblocks):
    """returns CSALI(t_b) = t_b * SALI_perp(t_b) at the end of every block"""
    s = x0.copy(); W = np.zeros((4, 2)); W[0, 0] = 1.0; W[2, 1] = 1.0
    proj_perp(s, W)
    out = np.empty(nblocks)
    for b in range(nblocks):
        y4_tan(s, W, dt, tau_steps)
        proj_perp(s, W)
        out[b] = sali2(W)*(b+1)*tau_steps*dt
    return out


ORBITS = {'R': 0.1, 'S': -0.0135, 'C': -0.015}   # x=0, py=0, px from E=1/8

def three_orbits(T=20000.0, dt=0.01, tau=100, sigmas=(1e-3, 3e-3, 1e-2)):
    nbk = int(T/(tau*dt)); W0 = np.zeros((4, 2)); W0[0, 0] = 1; W0[2, 1] = 1
    nodes, wts = gh_nodes(3); res = {}
    first = lambda a, th: (np.argmax(a < th)+1)*tau*dt if np.any(a < th) else None
    for k, y in ORBITS.items():
        x0 = np.array([0., y, px_from(y, 0.), 0.])
        res[k+'_sali'] = run_sali(x0, dt, tau, nbk, W0)
        print(k, 'SALI<1e-8 at t =', first(res[k+'_sali'], 1e-8))
        res[k+'_csali'] = run_csali(x0, dt, tau, nbk)
        print('   CSALI<1e-8 at t =', first(res[k+'_csali'], 1e-8), ' | SALI<1e-3:', first(res[k+'_sali'], 1e-3), ' CSALI<1e-3:', first(res[k+'_csali'], 1e-3))
        for s in sigmas:
            u = run_uali_iso(x0, dt, tau, nbk, W0.copy(), s, nodes, wts, 1)
            res[f'{k}_uali_{s:g}'] = u
            print(f'   UALI sigma={s:g}: <1e-8 at t =', first(u, 1e-8))
    np.savez('uali_three_orbits.npz', **res)
    return res

def grid(N=120, T=1000.0, dt=0.01, tau=100, sigmas=(1e-3, 1e-2)):
    nbk = int(T/(tau*dt))
    yv = np.linspace(-0.6, 0.8, N); pyv = np.linspace(-0.6, 0.6, N)
    YY, PP = np.meshgrid(yv, pyv)
    PX = np.array([px_from(a, b) for a, b in zip(YY.ravel(), PP.ravel())])
    nodes, wts = gh_nodes(3)
    out = scan(YY.ravel(), PP.ravel(), PX, dt, tau, nbk, np.array(sigmas), nodes, wts)
    np.savez('uali_grid.npz', yv=yv, pyv=pyv, out=out.reshape(N, N, -1), sig=np.array(sigmas), T=T)
    return out

if __name__ == '__main__':
    three_orbits()
    grid()
