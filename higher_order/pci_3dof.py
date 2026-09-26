"""PCI test on the 3-dof Hamiltonian H = sum w_i/2 (q_i^2+p_i^2) + q1^2 q2 + q1^2 q3."""
import numpy as np, numba as nb, math, time, sys
from csali_3dof import y4_tan, WW, gradV   # keep csali_3dof.py in the same folder

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

def labels(ref):
    return np.where(ref < 1e-8, 1, np.where(ref > 1e-4, 0, -1))

def feat(A, h, cols=(0, 1, 3, 4, 2)):      # PCI-2: -logSALI, -logSALI_perp, OFLI, MEGNO, FLI
    return np.c_[tuple(L(A[:, h, c]) if c in (0, 1) else A[:, h, c] for c in cols)]

if __name__ == '__main__':
    # Usage: python pci_3dof.py      (trains at E=0.07, tests held-out half and E=0.06, 0.09)
    dt, tau = 0.01, 500
    H = np.array([25, 50, 100, 200, 300, 500, 1000, 2000, 5000]); hidx = (H/5 - 1).astype(np.int64)
    W0 = np.random.default_rng(1).standard_normal((6, 3))
    data = {}
    for E, n in [(0.07, 3000), (0.06, 1500), (0.09, 1500)]:
        X = points(n, E); A, B, ref = run_all(X, W0, dt, tau, int(H[-1]/5), hidx, 2000, WW)
        lab = labels(ref); k = lab >= 0; data[E] = (A[k], B[k], lab[k])
        np.savez(f'pci3_E{E:g}.npz', X=X, A=A, B=B, ref=ref, H=H)
    A, B, lab = data[0.07]; rng = np.random.default_rng(0); tr = np.zeros(len(lab), bool)
    for c in (0, 1):
        ix = np.where(lab == c)[0]; rng.shuffle(ix); tr[ix[:len(ix)//2]] = True
    te = ~tr
    for E, (AA, BB, ll) in data.items():
        sel = te if E == 0.07 else np.ones(len(ll), bool)
        print(f'E={E}: n={sel.sum()} chaotic={ll[sel].mean():.2f}   AUROC  SALI / GALI3 / OFLI / PCI-2')
        for h, T in enumerate(H):
            m = logit_fit(feat(A, h)[tr], lab[tr])
            print(f'  T={T:5d}  {auc(ll[sel], L(AA[sel][:, h, 0])):.3f} {auc(ll[sel], L(BB[sel][:, h])):.3f} '
                  f'{auc(ll[sel], AA[sel][:, h, 3]):.3f} {auc(ll[sel], m(feat(AA, h)[sel])):.3f}')
