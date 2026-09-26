"""PCI test on the 3-dof Hamiltonian H = sum w_i/2 (q_i^2+p_i^2) + q1^2 q2 + q1^2 q3."""
import numpy as np, numba as nb, math, time, sys
from h3 import y4_tan, WW, gradV

@nb.njit(cache=True)
def proj3(s, v, w):
    q = s[:3]; p = s[3:]
    gv = gradV(q, w)
    g = np.concatenate((gv, w*p)); f = np.concatenate((w*p, -gv))
    g = g/np.sqrt(np.sum(g*g)); f = f - np.dot(f, g)*g; f = f/np.sqrt(np.sum(f*f))
    return v - np.dot(v, g)*g - np.dot(v, f)*f

@nb.njit(cache=True)
def sp(a, b):
    a = a/np.sqrt(np.sum(a*a)); b = b/np.sqrt(np.sum(b*b))
    return min(np.sqrt(np.sum((a+b)**2)), np.sqrt(np.sum((a-b)**2)))

@nb.njit(cache=True)
def engineA(x0, W0, dt, tau, nbk, hidx, w):
    s = x0.copy(); V = W0[:, :2].copy()
    for j in range(2):
        V[:, j] /= np.sqrt(np.sum(V[:, j]**2))
    out = np.full((hidx.size, 5), np.nan)
    L0 = 0.0; fli = -1e300; ofli = -1e300; lw = 0.0; S = 0.0; Yint = 0.0; t = 0.0; hk = 0
    for b in range(nbk):
        for k in range(tau):
            y4_tan(s, V, dt, 1, w); t += dt
            ln = L0 + 0.5*math.log(np.sum(V[:, 0]**2))
            S += t*(ln - lw); lw = ln; Yint += (2.0*S/t)*dt
        pa = proj3(s, V[:, 0].copy(), w); pb = proj3(s, V[:, 1].copy(), w)
        n0 = math.sqrt(np.sum(V[:, 0]**2)); n1 = math.sqrt(np.sum(V[:, 1]**2))
        fli = max(fli, L0 + math.log(n0)); ofli = max(ofli, L0 + math.log(math.sqrt(np.sum(pa*pa))))
        if hk < hidx.size and b == hidx[hk]:
            out[hk, 0] = sp(V[:, 0], V[:, 1]); out[hk, 1] = sp(pa, pb)
            out[hk, 2] = fli; out[hk, 3] = ofli; out[hk, 4] = Yint/t; hk += 1
        L0 += math.log(n0); V[:, 0] /= n0; V[:, 1] /= n1
    return out

@nb.njit(cache=True)
def engineB(x0, W0, dt, tau, nbk, hidx, w):
    s = x0.copy(); V = W0.copy(); out = np.full(hidx.size, np.nan); hk = 0
    for b in range(nbk):
        y4_tan(s, V, dt, tau, w)
        for j in range(3):
            V[:, j] /= np.sqrt(np.sum(V[:, j]**2))
        if hk < hidx.size and b == hidx[hk]:
            out[hk] = np.prod(np.linalg.svd(V)[1]); hk += 1
    return out

@nb.njit(cache=True)
def sali_ref(x0, W0, dt, tau, nbk, w):
    s = x0.copy(); V = W0[:, :2].copy(); v = 1.0
    for b in range(nbk):
        y4_tan(s, V, dt, tau, w)
        v = sp(V[:, 0], V[:, 1])
        for j in range(2):
            V[:, j] /= np.sqrt(np.sum(V[:, j]**2))
        if v < 1e-12:
            break
    return v

@nb.njit(parallel=True, cache=True)
def run_all(X, W0, dt, tau, nbk, hidx, nbref, w):
    n = X.shape[0]
    A = np.empty((n, hidx.size, 5)); B = np.empty((n, hidx.size)); ref = np.empty(n)
    for i in nb.prange(n):
        ref[i] = sali_ref(X[i], W0, dt, tau, nbref, w)
        A[i] = engineA(X[i], W0, dt, tau, nbk, hidx, w)
        B[i] = engineB(X[i], W0, dt, tau, nbk, hidx, w)
    return A, B, ref

def points(n, E, seed=7):
    rng = np.random.default_rng(seed); P = []
    a1, a2 = np.sqrt(2*E/WW[0]), np.sqrt(2*E/WW[1])
    while len(P) < n:
        p1, p2 = rng.uniform(-a1, a1), rng.uniform(-a2, a2)
        v = 2*E - WW[0]*p1**2 - WW[1]*p2**2
        if v > 1e-6:
            P.append((0, 0, 0, p1, p2, np.sqrt(v/WW[2])))
    return np.array(P, float)

if __name__ == '__main__':
    E = float(sys.argv[1]); n = int(sys.argv[2]); fn = sys.argv[3]
    dt, tau = 0.01, 500
    H = np.array([int(x) for x in sys.argv[4].split(',')]) if len(sys.argv) > 4 else np.array([25, 50, 100, 200, 300, 500, 1000]); hidx = (H/5 - 1).astype(np.int64)
    W0 = np.random.default_rng(1).standard_normal((6, 3))
    X = points(n, E); t0 = time.time()
    A, B, ref = run_all(X, W0, dt, tau, int(H[-1]/5), hidx, 2000, WW)
    print(fn, 'time', time.time()-t0, flush=True)
    np.savez(fn, X=X, A=A, B=B, ref=ref, H=H)
