"""Test bed for new/alternative indicators vs SALI on Henon-Heiles (E=1/8).
All indicators are read out at several horizons from ONE run per point.

Engine A (2 tangent vectors, same cost as SALI), initial vectors e_x, e_px:
  sali      : standard SALI (4D)
  salip     : SALI of the section projections  P_t w_i  (remove grad-H and flow
              components at READOUT only;  P_t Phi = P_t Phi P_0, so this is exact)
  fli, ofli : sup_t log|w_1| and sup_t log|P_t w_1| (cumulative through renormalisation)
  megno     : time-averaged MEGNO  <Y>  of w_1
  area      : log(|P w1||P w2| sin angle)  (section area; ~conserved -> sanity check)
Engine B (3 tangent vectors): gali3.
"""
import numpy as np, numba as nb, time, math
from uali import px_from

W1 = 1.0/(2.0-2.0**(1.0/3.0)); W0 = -2.0**(1.0/3.0)/(2.0-2.0**(1.0/3.0))

@nb.njit(cache=True, inline='always')
def lf_tan(s, V, h):
    x, y, px, py = s[0], s[1], s[2], s[3]
    a = 1+2*y; b = 2*x; d = 1-2*y
    for j in range(V.shape[1]):
        V[2, j] -= 0.5*h*(a*V[0, j] + b*V[1, j]); V[3, j] -= 0.5*h*(b*V[0, j] + d*V[1, j])
    px -= 0.5*h*(x + 2*x*y); py -= 0.5*h*(y + x*x - y*y)
    x += h*px; y += h*py
    for j in range(V.shape[1]):
        V[0, j] += h*V[2, j]; V[1, j] += h*V[3, j]
    a = 1+2*y; b = 2*x; d = 1-2*y
    for j in range(V.shape[1]):
        V[2, j] -= 0.5*h*(a*V[0, j] + b*V[1, j]); V[3, j] -= 0.5*h*(b*V[0, j] + d*V[1, j])
    px -= 0.5*h*(x + 2*x*y); py -= 0.5*h*(y + x*x - y*y)
    s[0] = x; s[1] = y; s[2] = px; s[3] = py

@nb.njit(cache=True)
def proj(s, v):
    x, y, px, py = s[0], s[1], s[2], s[3]
    g = np.array([x + 2*x*y, y + x*x - y*y, px, py]); g /= np.sqrt(np.sum(g*g))
    f = np.array([px, py, -(x + 2*x*y), -(y + x*x - y*y)])
    f -= np.dot(f, g)*g; f /= np.sqrt(np.sum(f*f))
    return v - np.dot(v, g)*g - np.dot(v, f)*f

@nb.njit(cache=True)
def sali_pair(a, b):
    a = a/np.sqrt(np.sum(a*a)); b = b/np.sqrt(np.sum(b*b))
    return min(np.sqrt(np.sum((a+b)**2)), np.sqrt(np.sum((a-b)**2)))

@nb.njit(cache=True)
def engineA(x0, dt, tau, nbk, hidx):
    """hidx: block indices (0-based) at which to record features."""
    s = x0.copy(); V = np.zeros((4, 2)); V[0, 0] = 1.0; V[2, 1] = 1.0
    out = np.full((hidx.size, 6), np.nan)
    L = np.zeros(2); fli = -1e300; ofli = -1e300
    lw = 0.0; S = 0.0; Yint = 0.0; t = 0.0; hk = 0
    for b in range(nbk):
        for k in range(tau):
            for c in (W1, W0, W1):
                lf_tan(s, V, c*dt)
            t += dt
            ln = L[0] + 0.5*math.log(V[0, 0]**2 + V[1, 0]**2 + V[2, 0]**2 + V[3, 0]**2)
            S += t*(ln - lw); lw = ln
            Yint += (2.0*S/t)*dt
        # readout
        pa = proj(s, V[:, 0]); pb = proj(s, V[:, 1])
        n0 = math.sqrt(np.sum(V[:, 0]**2)); n1 = math.sqrt(np.sum(V[:, 1]**2))
        fli = max(fli, L[0] + math.log(n0))
        ofli = max(ofli, L[0] + math.log(math.sqrt(np.sum(pa*pa))))
        if hk < hidx.size and b == hidx[hk]:
            na = math.sqrt(np.sum(pa*pa)); nbb = math.sqrt(np.sum(pb*pb))
            ea = pa/na; eb = pb/nbb; cth = abs(np.dot(ea, eb))
            area = L[0] + L[1] + math.log(na) + math.log(nbb) + 0.5*math.log(max(1e-300, 1 - cth*cth))
            out[hk, 0] = sali_pair(V[:, 0], V[:, 1]); out[hk, 1] = sali_pair(pa, pb)
            out[hk, 2] = fli; out[hk, 3] = ofli; out[hk, 4] = Yint/t; out[hk, 5] = area
            hk += 1
        L[0] += math.log(n0); L[1] += math.log(n1)
        V[:, 0] /= n0; V[:, 1] /= n1
    return out

@nb.njit(cache=True)
def engineB(x0, dt, tau, nbk, hidx):
    s = x0.copy()
    V = np.array([[1., 0, 0.3], [0, 1., 0.2], [0.1, 0, 1.], [0, 0.4, 0]])
    out = np.full(hidx.size, np.nan); hk = 0
    for b in range(nbk):
        for k in range(tau):
            for c in (W1, W0, W1):
                lf_tan(s, V, c*dt)
        for j in range(3):
            V[:, j] /= np.sqrt(np.sum(V[:, j]**2))
        if hk < hidx.size and b == hidx[hk]:
            out[hk] = np.prod(np.linalg.svd(V)[1]); hk += 1
    return out

@nb.njit(cache=True)
def sali_ref(x0, dt, tau, nbk):
    s = x0.copy(); V = np.zeros((4, 2)); V[0, 0] = 1.0; V[2, 1] = 1.0
    v = 1.0
    for b in range(nbk):
        for k in range(tau):
            for c in (W1, W0, W1):
                lf_tan(s, V, c*dt)
        v = sali_pair(V[:, 0], V[:, 1])
        V[:, 0] /= np.sqrt(np.sum(V[:, 0]**2)); V[:, 1] /= np.sqrt(np.sum(V[:, 1]**2))
        if v < 1e-12:
            break
    return v

@nb.njit(parallel=True, cache=True)
def run_all(X, dt, tau, nbk, hidx, nbref):
    n = X.shape[0]
    A = np.empty((n, hidx.size, 6)); B = np.empty((n, hidx.size)); ref = np.empty(n)
    for i in nb.prange(n):
        ref[i] = sali_ref(X[i], dt, tau, nbref)
        A[i] = engineA(X[i], dt, tau, nbk, hidx)
        B[i] = engineB(X[i], dt, tau, nbk, hidx)
    return A, B, ref

if __name__ == '__main__':
    import sys
    E = float(sys.argv[2]) if len(sys.argv) > 2 else 0.125; n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    rng = np.random.default_rng(7); pts = []
    while len(pts) < n:
        y, py = rng.uniform(-.7, 1.0), rng.uniform(-.7, .7); px = px_from(y, py, E)
        if np.isfinite(px): pts.append((0., y, px, py))
    X = np.array(pts); dt = 0.01; tau = 500           # tau = 5 time units, like the user's run
    H = np.array([25, 50, 100, 150, 200, 300, 500]); hidx = (H/5 - 1).astype(np.int64)
    t0 = time.time()
    A, B, ref = run_all(X, dt, tau, int(H[-1]/5), hidx, 2000)
    print('run', time.time()-t0)
    # timings per point to T=500 (single thread)
    x = X[0]; engineA(x, dt, tau, 2, hidx[:1]); engineB(x, dt, tau, 2, hidx[:1]); sali_ref(x, dt, tau, 2)
    tt = {}
    for name, f in [('A', lambda: engineA(x, dt, tau, 100, hidx)), ('B', lambda: engineB(x, dt, tau, 100, hidx)),
                    ('sali', lambda: sali_ref(np.array([0., 0.1, px_from(0.1, 0.), 0.]), dt, tau, 100))]:
        t0 = time.perf_counter(); [f() for _ in range(5)]; tt[name] = (time.perf_counter()-t0)/5
    print('wall per point to T=500:', tt)
    np.savez(sys.argv[3] if len(sys.argv) > 3 else 'newind.npz', X=X, A=A, B=B, ref=ref, H=H, tA=tt['A'], tB=tt['B'], tS=tt['sali'])
