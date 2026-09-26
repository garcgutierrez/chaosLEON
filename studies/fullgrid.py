"""Full-grid, one-trajectory-per-node chaos map (grid-as-ensemble) vs SALI map.
Every grid node is integrated once; stretching and folding of each node's
'overlapping kernel' are read from its grid neighbours (finite differences)."""
import numpy as np, numba as nb, math, time, sys
from newind import W1, W0, proj, sali_ref, lf_tan, sali_pair, px_from
from fold import wedge
from gridnb import lf

@nb.njit(parallel=True, cache=True)
def advance(S, valid, dt, n):
    for i in nb.prange(S.shape[0]):
        if valid[i]:
            s = S[i]
            for _ in range(n):
                lf(s, W1*dt); lf(s, W0*dt); lf(s, W1*dt)

@nb.njit(parallel=True, cache=True)
def fd_update(S, ok, Ny, Np, hy, hp, sup_fold, sup_ofli):
    for j in nb.prange(Np):
        for i in range(Ny):
            idx = j*Ny + i
            if not ok[idx]:
                continue
            c = S[idx]; yp = S[idx+1]; ym = S[idx-1]; pp = S[idx+Ny]; pm = S[idx-Ny]
            Dy = (yp - ym)/(2*hy); Dp = (pp - pm)/(2*hp)
            Sy = (yp + ym - 2*c)/(hy*hy); Sp = (pp + pm - 2*c)/(hp*hp)
            pDy = proj(c, Dy); pDp = proj(c, Dp)
            ny = math.sqrt(np.sum(pDy*pDy)); npp = math.sqrt(np.sum(pDp*pDp))
            by = hy*wedge(pDy, proj(c, Sy))/(ny*ny); bp = hp*wedge(pDp, proj(c, Sp))/(npp*npp)
            sup_fold[idx] = max(sup_fold[idx], math.log(max(by, bp, 1e-300)))
            sup_ofli[idx] = max(sup_ofli[idx], math.log(max(ny, npp)))

@nb.njit(parallel=True, cache=True)
def sali_grid(X, valid, dt, tau, nbk):
    out = np.full(X.shape[0], np.nan)
    for i in nb.prange(X.shape[0]):
        if valid[i]:
            s = X[i].copy(); V = np.zeros((4, 2)); V[0, 0] = 1.0; V[2, 1] = 1.0
            for b in range(nbk):
                for k in range(tau):
                    for c in (W1, W0, W1):
                        lf_tan(s, V, c*dt)
                out[i] = sali_pair(V[:, 0], V[:, 1])
                V[:, 0] /= np.sqrt(np.sum(V[:, 0]**2)); V[:, 1] /= np.sqrt(np.sum(V[:, 1]**2))
    return out

@nb.njit(parallel=True, cache=True)
def ref_grid(X, valid, stride):
    out = np.full(X.shape[0], np.nan)
    for i in nb.prange(X.shape[0]):
        if valid[i] and (i % stride == 0):
            out[i] = sali_ref(X[i], 0.01, 500, 2000)
    return out

if __name__ == '__main__':
    N = int(sys.argv[1]); T = float(sys.argv[2])
    yv = np.linspace(-0.6, 0.8, N); pv = np.linspace(-0.6, 0.6, N); hy, hp = yv[1]-yv[0], pv[1]-pv[0]
    X = np.zeros((N*N, 4)); valid = np.zeros(N*N, np.bool_)
    for j in range(N):
        for i in range(N):
            px = px_from(yv[i], pv[j])
            if np.isfinite(px) and px > 1e-3:
                X[j*N+i] = (0., yv[i], px, pv[j]); valid[j*N+i] = True
    ok = valid.copy().reshape(N, N)
    ok[:, 0] = ok[:, -1] = ok[0, :] = ok[-1, :] = False
    v2 = valid.reshape(N, N)
    ok[1:-1, 1:-1] &= v2[1:-1, 2:] & v2[1:-1, :-2] & v2[2:, 1:-1] & v2[:-2, 1:-1]
    ok = ok.ravel()
    # warm-up
    S = X.copy(); advance(S[:10], valid[:10], 0.01, 1)
    sf = np.full(N*N, -1e300); so = np.full(N*N, -1e300)
    fd_update(S, np.zeros(N*N, np.bool_), N, N, hy, hp, sf, so); sali_grid(X[:10], valid[:10], 0.01, 1, 1)
    t0 = time.time(); S = X.copy()
    for b in range(int(T/5)):
        advance(S, valid, 0.01, 500); fd_update(S, ok, N, N, hy, hp, sf, so)
    t_fd = time.time() - t0
    t0 = time.time(); sal = sali_grid(X, valid, 0.01, 500, int(T/5)); t_sali = time.time() - t0
    print(f'grid {N}x{N}, T={T}: valid nodes {valid.sum()}, grid-FD map {t_fd:.1f}s, SALI map {t_sali:.1f}s', flush=True)
    t0 = time.time(); ref = ref_grid(X, ok, 7); print('reference SALI(1e4) on every 7th node', time.time()-t0, flush=True)
    np.savez(f'fullgrid_{N}_T{int(T)}.npz', yv=yv, pv=pv, ok=ok, valid=valid, fold=sf, ofli=so, sali=sal, ref=ref, t_fd=t_fd, t_sali=t_sali)
