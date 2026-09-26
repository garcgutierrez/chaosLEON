# -*- coding: utf-8 -*-
# Usage: python csali_3dof.py 0.07   (energy); writes b3_E<E>.npz with threshold-crossing times
# Note: the SALI paper orbit (0,0,0,0.14142,0.11892,0.29428) has H = 0.095, not 0.09.
"""3-dof Hamiltonian of Antonopoulos, Manos & Skokos (2004), eq. (3):
H = sum_i w_i/2 (q_i^2 + p_i^2) + q1^2 q2 + q1^2 q3
Indicators: SALI, GALI_3 (6D), SALI_perp and GALI_3_perp (vectors projected on the
energy shell modulo the flow, i.e. the 4D linearised Poincare map)."""
import numpy as np, numba as nb

WW = np.array([1.0, 1.4142, 1.7321])
Y1 = 1.0/(2.0-2.0**(1.0/3.0)); Y0 = -2.0**(1.0/3.0)/(2.0-2.0**(1.0/3.0))

@nb.njit(cache=True)
def gradV(q, w):
    return np.array([w[0]*q[0] + 2*q[0]*q[1] + 2*q[0]*q[2], w[1]*q[1] + q[0]**2, w[2]*q[2] + q[0]**2])

@nb.njit(cache=True)
def hessV(q, w):
    H = np.zeros((3, 3))
    H[0, 0] = w[0] + 2*q[1] + 2*q[2]; H[0, 1] = H[1, 0] = 2*q[0]; H[0, 2] = H[2, 0] = 2*q[0]
    H[1, 1] = w[1]; H[2, 2] = w[2]
    return H

@nb.njit(cache=True)
def Hval(s, w):
    q = s[:3]; p = s[3:]
    return 0.5*np.sum(w*(q*q + p*p)) + q[0]**2*q[1] + q[0]**2*q[2]

@nb.njit(cache=True)
def y4_tan(s, W, dt, nsteps, w):
    q0, q1, q2, p0, p1, p2 = s[0], s[1], s[2], s[3], s[4], s[5]
    w0, w1, w2 = w[0], w[1], w[2]
    k = W.shape[1]
    for _ in range(nsteps):
        for c in (Y1, Y0, Y1):
            h = 0.5*c*dt
            for half in range(2):
                a = w0 + 2*q1 + 2*q2; b = 2*q0
                for j in range(k):
                    d0 = W[0, j]; d1 = W[1, j]; d2 = W[2, j]
                    W[3, j] -= h*(a*d0 + b*d1 + b*d2)
                    W[4, j] -= h*(b*d0 + w1*d1)
                    W[5, j] -= h*(b*d0 + w2*d2)
                p0 -= h*(w0*q0 + 2*q0*q1 + 2*q0*q2)
                p1 -= h*(w1*q1 + q0*q0)
                p2 -= h*(w2*q2 + q0*q0)
                if half == 0:
                    q0 += 2*h*w0*p0; q1 += 2*h*w1*p1; q2 += 2*h*w2*p2
                    for j in range(k):
                        W[0, j] += 2*h*w0*W[3, j]; W[1, j] += 2*h*w1*W[4, j]; W[2, j] += 2*h*w2*W[5, j]
    s[0] = q0; s[1] = q1; s[2] = q2; s[3] = p0; s[4] = p1; s[5] = p2

@nb.njit(cache=True)
def normalize_cols(W):
    for j in range(W.shape[1]):
        W[:, j] /= np.sqrt(np.sum(W[:, j]**2))

@nb.njit(cache=True)
def proj_perp(s, W, w):
    q = s[:3]; p = s[3:]
    gv = gradV(q, w)
    g = np.concatenate((gv, w*p)); f = np.concatenate((w*p, -gv))
    g = g/np.sqrt(np.sum(g*g)); f = f - np.dot(f, g)*g; f = f/np.sqrt(np.sum(f*f))
    for j in range(W.shape[1]):
        v = W[:, j].copy()
        v = v - np.dot(v, g)*g; v = v - np.dot(v, f)*f
        W[:, j] = v

@nb.njit(cache=True)
def sali(W):
    e1 = W[:, 0]; e2 = W[:, 1]
    return min(np.sqrt(np.sum((e1+e2)**2)), np.sqrt(np.sum((e1-e2)**2)))

@nb.njit(cache=True)
def gali(W):
    sv = np.linalg.svd(W)[1]
    return np.prod(sv)

@nb.njit(cache=True)
def run_all(x0, W0, dt, tau, nbk, w, perp):
    """returns (nbk,2): SALI of first two columns, GALI_k of all columns (normalised)."""
    s = x0.copy(); W = W0.copy()
    if perp:
        proj_perp(s, W, w)
    normalize_cols(W)
    out = np.empty((nbk, 2))
    for b in range(nbk):
        y4_tan(s, W, dt, tau, w)
        if perp:
            proj_perp(s, W, w)
        normalize_cols(W)
        out[b, 0] = sali(W[:, :2]); out[b, 1] = gali(W)
    return out

def rand_W0(k, seed=1):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((6, k))


# ---------------- grid benchmark (threshold-crossing times) ----------------
TH = np.array([1e-3, 1e-4, 1e-6, 1e-8])

@nb.njit(cache=True)
def cross(x0, W0, dt, tau, nbk, w, perp, mode, pw, stop_th):
    """mode 0: SALI of cols 0,1 ; mode 1: GALI_k of all cols. index * t^pw."""
    s = x0.copy(); W = W0.copy()
    if perp:
        proj_perp(s, W, w)
    normalize_cols(W)
    tc = np.full(4, np.nan)
    for b in range(nbk):
        y4_tan(s, W, dt, tau, w)
        if perp:
            proj_perp(s, W, w)
        normalize_cols(W)
        t = (b+1)*tau*dt
        v = (sali(W) if mode == 0 else gali(W))*t**pw
        for j in range(4):
            if np.isnan(tc[j]) and v < TH[j]:
                tc[j] = t
        if v < stop_th:
            break
    return tc

@nb.njit(parallel=True, cache=True)
def scan(P1, P2, P3, W0, dt, tau, nbk, w, perp, mode, pw, stop_th):
    out = np.full((P1.size, 4), np.nan)
    for i in nb.prange(P1.size):
        if not np.isnan(P3[i]):
            x0 = np.array([0.0, 0.0, 0.0, P1[i], P2[i], P3[i]])
            out[i] = cross(x0, W0, dt, tau, nbk, w, perp, mode, pw, stop_th)
    return out

if __name__ == '__main__':
    import sys
    E = float(sys.argv[1]) if len(sys.argv) > 1 else 0.09; N = 60; dt = 0.01; tau = 100
    FN = f'b3_E{E:g}.npz'
    p1v = np.linspace(-np.sqrt(2*E/WW[0]), np.sqrt(2*E/WW[0]), N)
    p2v = np.linspace(-np.sqrt(2*E/WW[1]), np.sqrt(2*E/WW[1]), N)
    A, B = np.meshgrid(p1v, p2v); P1, P2 = A.ravel(), B.ravel()
    v = 2*E - WW[0]*P1**2 - WW[1]*P2**2
    P3 = np.where(v > 1e-6, np.sqrt(np.maximum(v, 0)/WW[2]), np.nan)
    W2 = rand_W0(2); W3 = rand_W0(3)
    res = {'P1': P1, 'P2': P2, 'P3': P3}
    # cost per unit time (regular-ish orbit, no early stop)
    xr = np.array([0, 0, 0, 0.3, 0.05, np.sqrt((2*E - WW[0]*0.09 - WW[1]*0.0025)/WW[2])])
    cost = {}
    for name, W0 in [('k2', W2), ('k3', W3)]:
        for perp in (False, True):
            cross(xr, W0, dt, tau, 2, WW, perp, 0, 0.0, -1.0)
            t0 = time.perf_counter(); cross(xr, W0, dt, tau, 500, WW, perp, 0, 0.0, -1.0)
            cost[f'{name}_{int(perp)}'] = (time.perf_counter()-t0)/500
    print('cost/time unit', cost, flush=True)
    res['cost'] = np.array([cost['k2_0'], cost['k3_0'], cost['k2_1'], cost['k3_1']])
    t0 = time.time()
    res['truth_sali'] = scan(P1, P2, P3, W2, dt, tau, 10000, WW, False, 0, 0.0, 1e-8)
    print('truth sali', time.time()-t0, flush=True); np.savez(FN, **res)
    res['truth_gali'] = scan(P1, P2, P3, W3, dt, tau, 10000, WW, False, 1, 0.0, 1e-12)
    print('truth gali', time.time()-t0, flush=True); np.savez(FN, **res)
    T = 3000
    cfg = [('SALI', W2, False, 0, 0.0), ('GALI3', W3, False, 1, 0.0), ('SALIperp', W2, True, 0, 0.0),
           ('GALI3perp_p0', W3, True, 1, 0.0), ('GALI3perp_p1', W3, True, 1, 1.0), ('GALI3perp_p2', W3, True, 1, 2.0)]
    for name, W0, perp, mode, pw in cfg:
        t0 = time.time()
        res[name] = scan(P1, P2, P3, W0, dt, tau, T, WW, perp, mode, pw, 1e-8)
        print(name, time.time()-t0, flush=True); np.savez(FN, **res)
