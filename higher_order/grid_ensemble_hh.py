"""Grid-as-ensemble chaos map for Henon-Heiles (PSS x=0, E=1/8), compared with a SALI map.
Usage: python grid_ensemble_hh.py 600 200      (grid N x N, horizon T)
Writes fullgrid_<N>_T<T>.npz with: fold (sup finite-size bending), ofli (sup stretch),
sali (T), ref (SALI at 1e4 on every 7th node) and the two wall times.
Every grid node is integrated once; stretching and folding of each node's
'overlapping kernel' are read from its grid neighbours (finite differences)."""
import numpy as np, numba as nb, math, time, sys
import numpy.polynomial.chebyshev as C
from new_indicators_hh import W1, W0, proj, sali_ref, lf_tan, sali_pair, px_from   # same folder
from fold_indicator_hh import wedge                                                 # same folder

@nb.njit(cache=True, inline='always')
def lf(s, h):
    x, y, px, py = s[0], s[1], s[2], s[3]
    px -= 0.5*h*(x + 2*x*y); py -= 0.5*h*(y + x*x - y*y)
    x += h*px; y += h*py
    px -= 0.5*h*(x + 2*x*y); py -= 0.5*h*(y + x*x - y*y)
    s[0] = x; s[1] = y; s[2] = px; s[3] = py

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

# ---- optional vectorised Picard-Chebyshev (MCPI) propagator for the whole grid ----
def cheb_integration_matrix(N):
    tau = -np.cos(np.pi*np.arange(N+1)/N)                 # ascending CGL nodes on [-1,1]
    V = C.chebvander(tau, N)                               # V[i,k] = T_k(tau_i)
    Vint = np.empty_like(V)
    for k in range(N+1):
        e = np.zeros(N+1); e[k] = 1
        Vint[:, k] = C.chebval(tau, C.chebint(e, lbnd=-1))
    return tau, Vint @ np.linalg.inv(V)

def rhs(Z):
    x, y, px, py = Z[..., 0], Z[..., 1], Z[..., 2], Z[..., 3]
    return np.stack((px, py, -(x + 2*x*y), -(y + x*x - y*y)), -1)

def mcpi_segment(z0, dt_seg, I, guess=None, tol=1e-13, maxit=60):
    """z0: (M,4). Returns Z at all nodes (N+1,M,4), number of Picard sweeps."""
    n1 = I.shape[0]; M = z0.shape[0]
    Z = np.broadcast_to(z0, (n1, M, 4)).copy() if guess is None else guess
    for it in range(maxit):
        F = rhs(Z).reshape(n1, M*4)
        Zn = z0[None] + 0.5*dt_seg*(I @ F).reshape(n1, M, 4)
        err = np.max(np.abs(Zn - Z)); Z = Zn
        if err < tol:
            break
    return Z, it+1

def propagate(z0, T, dt_seg, N):
    tau, I = cheb_integration_matrix(N)
    z = z0.copy(); its = []
    for s in range(int(round(T/dt_seg))):
        Z, it = mcpi_segment(z, dt_seg, I); z = Z[-1].copy(); its.append(it)
    return z, np.mean(its)


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
