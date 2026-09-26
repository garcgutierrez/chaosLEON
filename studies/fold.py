"""Second-order (curvature) fold indicator, a cheap Taylor model of UFR.

A material segment x0 + s u (|s| <= sigma) is mapped to
    x(t,s) = phi_t(x0) + s w(t) + s^2/2 w2(t) + O(s^3),
with w = D phi u (first variation) and w2 = D^2 phi[u,u] (second variation).
Its bending angle across the segment (curvature x length) is
    beta(t) = sigma * |P w  ^  P w2| / |P w|^2 ,      P = section projection,
and a fold (hull ratio > 1) appears when beta ~ O(1).  log beta = log sigma + FI(t),
so the RANKING is sigma-independent; sigma only sets the fold threshold.
First and second variations use the exact discrete variation of the Yoshida
leapfrog, so they are consistent with the orbit integrator.
"""
import numpy as np, numba as nb, math

W1 = 1.0/(2.0-2.0**(1.0/3.0)); W0 = -2.0**(1.0/3.0)/(2.0-2.0**(1.0/3.0))

@nb.njit(cache=True, inline='always')
def kick(s, w, w2, h):
    x, y = s[0], s[1]
    # force F = (-(x+2xy), -(y+x^2-y^2)); DF and D2F
    a = -(1+2*y); b = -2*x; d = -(1-2*y)
    wx, wy = w[0], w[1]
    d2x = -4*wx*wy; d2y = -2*wx*wx + 2*wy*wy
    w2[2] += h*(a*w2[0] + b*w2[1] + d2x); w2[3] += h*(b*w2[0] + d*w2[1] + d2y)
    w[2] += h*(a*wx + b*wy); w[3] += h*(b*wx + d*wy)
    s[2] += h*(-(x + 2*x*y)); s[3] += h*(-(y + x*x - y*y))

@nb.njit(cache=True, inline='always')
def drift(s, w, w2, h):
    s[0] += h*s[2]; s[1] += h*s[3]
    w[0] += h*w[2]; w[1] += h*w[3]
    w2[0] += h*w2[2]; w2[1] += h*w2[3]

@nb.njit(cache=True)
def y4_2(s, w, w2, dt, n):
    for _ in range(n):
        for c in (W1, W0, W1):
            h = c*dt
            kick(s, w, w2, 0.5*h); drift(s, w, w2, h); kick(s, w, w2, 0.5*h)

@nb.njit(cache=True)
def proj(s, v):
    x, y, px, py = s[0], s[1], s[2], s[3]
    g = np.array([x + 2*x*y, y + x*x - y*y, px, py]); g /= np.sqrt(np.sum(g*g))
    f = np.array([px, py, -(x + 2*x*y), -(y + x*x - y*y)])
    f -= np.dot(f, g)*g; f /= np.sqrt(np.sum(f*f))
    return v - np.dot(v, g)*g - np.dot(v, f)*f

@nb.njit(cache=True)
def wedge(a, b):
    na = np.sqrt(np.sum(a*a)); nb_ = np.sqrt(np.sum(b*b))
    c = np.dot(a, b)/(na*nb_)
    return na*nb_*math.sqrt(max(0.0, 1 - c*c))

@nb.njit(cache=True)
def fold_run(x0, u0, dt, tau, nbk, hidx):
    """returns per horizon: FI = log(|Pw^Pw2|/|Pw|^2) (true scale), sup FI, OFLI"""
    s = x0.copy(); w = proj(s, u0.copy()); w /= np.sqrt(np.sum(w*w)); w2 = np.zeros(4)
    Lc = 0.0; supfi = -1e300; ofli = -1e300; hk = 0
    out = np.full((hidx.size, 3), np.nan)
    for b in range(nbk):
        y4_2(s, w, w2, dt, tau)
        pw = proj(s, w); pw2 = proj(s, w2)
        npw = math.sqrt(np.sum(pw*pw))
        fi = Lc + math.log(max(wedge(pw, pw2), 1e-300)) - 2*math.log(npw)
        supfi = max(supfi, fi); ofli = max(ofli, Lc + math.log(npw))
        if hk < hidx.size and b == hidx[hk]:
            out[hk, 0] = fi; out[hk, 1] = supfi; out[hk, 2] = ofli; hk += 1
        n = math.sqrt(np.sum(w*w)); Lc += math.log(n); w /= n; w2 /= n*n
    return out

@nb.njit(parallel=True, cache=True)
def fold_all(X, u0, dt, tau, nbk, hidx):
    out = np.empty((X.shape[0], hidx.size, 3))
    for i in nb.prange(X.shape[0]):
        out[i] = fold_run(X[i], u0, dt, tau, nbk, hidx)
    return out

@nb.njit(cache=True)
def fold_run2(x0, u0, dt, tau, nbk, hidx):
    """as fold_run, plus OFLI2 = sup log|P(w + w2/2)| (Barrio) for comparison."""
    s = x0.copy(); w = proj(s, u0.copy()); w /= np.sqrt(np.sum(w*w)); w2 = np.zeros(4)
    Lc = 0.0; supfi = -1e300; ofli = -1e300; ofli2 = -1e300; hk = 0
    out = np.full((hidx.size, 4), np.nan)
    for b in range(nbk):
        y4_2(s, w, w2, dt, tau)
        pw = proj(s, w); pw2 = proj(s, w2)
        npw = math.sqrt(np.sum(pw*pw))
        fi = Lc + math.log(max(wedge(pw, pw2), 1e-300)) - 2*math.log(npw)
        supfi = max(supfi, fi); ofli = max(ofli, Lc + math.log(npw))
        v = pw + 0.5*math.exp(min(Lc, 600.0))*pw2
        ofli2 = max(ofli2, Lc + math.log(math.sqrt(np.sum(v*v))))
        if hk < hidx.size and b == hidx[hk]:
            out[hk, 0] = fi; out[hk, 1] = supfi; out[hk, 2] = ofli; out[hk, 3] = ofli2; hk += 1
        n = math.sqrt(np.sum(w*w)); Lc += math.log(n); w /= n; w2 /= n*n
    return out

@nb.njit(parallel=True, cache=True)
def fold_all2(X, u0, dt, tau, nbk, hidx):
    out = np.empty((X.shape[0], hidx.size, 4))
    for i in nb.prange(X.shape[0]):
        out[i] = fold_run2(X[i], u0, dt, tau, nbk, hidx)
    return out
