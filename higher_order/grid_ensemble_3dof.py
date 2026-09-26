"""Usage: python grid_ensemble_3dof.py 0.07 1000 1500 out.npz   (E, grid N, sampled nodes, output)
Grid-as-ensemble indicators for the 3-dof Hamiltonian
H = sum w_i/2 (q_i^2+p_i^2) + q1^2 q2 + q1^2 q3, grid on q=0 in (p1,p2), p3>0 from H=E.
Each sampled node is integrated with its 4 grid neighbours (p1+-h1, p2+-h2); in a
full-grid run those are other grid nodes, so the cost is one trajectory per node."""
import numpy as np, numba as nb, math, sys, time
from csali_3dof import WW, Y1, Y0, gradV                        # same folder
from pci_3dof import proj3, sp, engineA, engineB, sali_ref      # same folder
from fold_indicator_hh import wedge                             # same folder

@nb.njit(cache=True)
def y4_orbits(S, dt, n, w):
    for _ in range(n):
        for c in (Y1, Y0, Y1):
            h = 0.5*c*dt
            for k in range(S.shape[0]):
                q0, q1, q2, p0, p1, p2 = S[k, 0], S[k, 1], S[k, 2], S[k, 3], S[k, 4], S[k, 5]
                p0 -= h*(w[0]*q0 + 2*q0*q1 + 2*q0*q2); p1 -= h*(w[1]*q1 + q0*q0); p2 -= h*(w[2]*q2 + q0*q0)
                q0 += 2*h*w[0]*p0; q1 += 2*h*w[1]*p1; q2 += 2*h*w[2]*p2
                p0 -= h*(w[0]*q0 + 2*q0*q1 + 2*q0*q2); p1 -= h*(w[1]*q1 + q0*q0); p2 -= h*(w[2]*q2 + q0*q0)
                S[k, 0] = q0; S[k, 1] = q1; S[k, 2] = q2; S[k, 3] = p0; S[k, 4] = p1; S[k, 5] = p2

@nb.njit(cache=True)
def node(S0, h1, h2, dt, tau, nbk, hidx, w):
    S = S0.copy(); out = np.full((hidx.size, 5), np.nan); so = -1e300; sf = -1e300; s2 = -1e300; s2h = -1e300; hk = 0
    for b in range(nbk):
        y4_orbits(S, dt, tau, w)
        c = S[0]
        D1 = (S[1] - S[2])/(2*h1); D2 = (S[3] - S[4])/(2*h2)
        C1 = (S[1] + S[2] - 2*c)/(h1*h1); C2 = (S[3] + S[4] - 2*c)/(h2*h2)
        a1 = proj3(c, D1, w); a2 = proj3(c, D2, w)
        n1 = math.sqrt(np.sum(a1*a1)); n2 = math.sqrt(np.sum(a2*a2))
        so = max(so, math.log(max(n1, n2)))
        b1 = h1*wedge(a1, proj3(c, C1, w))/(n1*n1); b2 = h2*wedge(a2, proj3(c, C2, w))/(n2*n2)
        sf = max(sf, math.log(max(b1, b2, 1e-300)))
        o1 = proj3(c, D1 + 0.5*C1, w); o2 = proj3(c, D2 + 0.5*C2, w)
        s2 = max(s2, math.log(max(math.sqrt(np.sum(o1*o1)), math.sqrt(np.sum(o2*o2)))))
        f1 = proj3(c, D1 + 0.5*h1*C1, w); f2 = proj3(c, D2 + 0.5*h2*C2, w)
        s2h = max(s2h, math.log(max(math.sqrt(np.sum(f1*f1)), math.sqrt(np.sum(f2*f2)))))
        if hk < hidx.size and b == hidx[hk]:
            out[hk, 0] = so; out[hk, 1] = sp(a1, a2); out[hk, 2] = sf; out[hk, 3] = s2; out[hk, 4] = s2h; hk += 1
    return out

@nb.njit(parallel=True, cache=True)
def run(S0all, W0, h1, h2, dt, tau, nbk, hidx, w):
    n = S0all.shape[0]
    F = np.empty((n, hidx.size, 5)); A = np.empty((n, hidx.size, 5)); B = np.empty((n, hidx.size)); ref = np.empty(n)
    for i in nb.prange(n):
        F[i] = node(S0all[i], h1, h2, dt, tau, nbk, hidx, w)
        A[i] = engineA(S0all[i, 0], W0, dt, tau, nbk, hidx, w)
        B[i] = engineB(S0all[i, 0], W0, dt, tau, nbk, hidx, w)
        ref[i] = sali_ref(S0all[i, 0], W0, dt, tau, 2000, w)
    return F, A, B, ref

def sample(E, Ngrid, nsamp, seed=5):
    a1, a2 = np.sqrt(2*E/WW[0]), np.sqrt(2*E/WW[1])
    p1v = np.linspace(-a1, a1, Ngrid); p2v = np.linspace(-a2, a2, Ngrid); h1, h2 = p1v[1]-p1v[0], p2v[1]-p2v[0]
    rng = np.random.default_rng(seed); S0 = []
    while len(S0) < nsamp:
        i, j = rng.integers(1, Ngrid-1, 2); st = []
        for p1, p2 in ((p1v[i], p2v[j]), (p1v[i]+h1, p2v[j]), (p1v[i]-h1, p2v[j]), (p1v[i], p2v[j]+h2), (p1v[i], p2v[j]-h2)):
            v = 2*E - WW[0]*p1**2 - WW[1]*p2**2
            if v <= 1e-6: break
            st.append((0, 0, 0, p1, p2, np.sqrt(v/WW[2])))
        if len(st) == 5: S0.append(st)
    return np.array(S0, float), h1, h2

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


if __name__ == '__main__':
    E, Ngrid, nsamp, fn = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    H = np.array([100, 200, 500, 1000, 2000]); hidx = (H/5 - 1).astype(np.int64)
    W0 = np.random.default_rng(1).standard_normal((6, 3))
    S0, h1, h2 = sample(E, Ngrid, nsamp); t0 = time.time()
    F, A, B, ref = run(S0, W0, h1, h2, 0.01, 500, int(H[-1]/5), hidx, WW)
    print(fn, f'E={E} grid {Ngrid} h=({h1:.4f},{h2:.4f}) time {time.time()-t0:.0f}s', flush=True)

    # second-order tangent comparison (OFLI2 and sup bending) along the grid p1 direction
    U = S0[:, 1] - S0[:, 2]; U /= np.linalg.norm(U, axis=1)[:, None]
    Fo = fold3_all(S0[:, 0].copy(), U, 0.01, 500, int(H[-1]/5), hidx, WW)
    np.savez(fn, S0=S0, F=F, A=A, B=B, ref=ref, H=H, h1=h1, h2=h2, Fo=Fo)
    from new_indicators_hh import auc, L
    lab = np.where(ref < 1e-8, 1, np.where(ref > 1e-4, 0, -1)); k = lab >= 0; y = lab[k]
    print('   T   SALI  GALI3  OFLI  OFLI2  supFI  grid-fold  grid-OFLI2')
    for h, T in enumerate(H):
        print(f'{T:5d}  {auc(y, L(A[k, h, 0])):.3f}  {auc(y, L(B[k, h])):.3f}  {auc(y, A[k, h, 3]):.3f}  {auc(y, Fo[k, h, 1]):.3f}  {auc(y, Fo[k, h, 0]):.3f}  {auc(y, F[k, h, 2]):.3f}  {auc(y, F[k, h, 3]):.3f}')
