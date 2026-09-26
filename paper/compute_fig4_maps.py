"""Data for Fig. 4: complete Henon-Heiles chaos maps (E = 1/8, section x = 0, 600^2 grid in (y, p_y), T = 200).

Maps: SALI(T) (two deviation vectors e_x, e_px), and the grid-as-ensemble indicators computed from the orbits of the
grid nodes alone: stretching S_h, bending B_h and OFLI2_h (five-point stencil) and LEON_h (nine-point stencil,
fourth-order bending, Sec. IV D).  Reference labels: SALI(t = 1e4) on every seventh admissible node.
Wall-clock times of the SALI map and of the grid maps are measured with NTHREADS threads.
Usage:  python compute_fig4_maps.py            -> ../data/fig4_maps.npz
"""
import numpy as np, numba as nb, math, time, os

E = 0.125; N = 600; T = 200.0; DT = 0.01; TAU = 500; TREF = 1e4; NTHREADS = 2
W1 = 1.0/(2.0 - 2.0**(1/3)); W0 = -2.0**(1/3)/(2.0 - 2.0**(1/3))


@nb.njit(cache=True, inline='always')
def kick(s, h):
    x, y = s[0], s[1]; s[2] -= h*(x + 2*x*y); s[3] -= h*(y + x*x - y*y)


@nb.njit(cache=True)
def orbit(s, n):
    for _ in range(n):
        for c in (W1, W0, W1):
            h = c*DT; kick(s, 0.5*h); s[0] += h*s[2]; s[1] += h*s[3]; kick(s, 0.5*h)


@nb.njit(cache=True)
def tangent(s, V, n):
    for _ in range(n):
        for c in (W1, W0, W1):
            h = c*DT
            for half in range(2):
                A = 1 + 2*s[1]; B = 2*s[0]; D = 1 - 2*s[1]
                for j in range(V.shape[1]):
                    V[2, j] -= 0.5*h*(A*V[0, j] + B*V[1, j]); V[3, j] -= 0.5*h*(B*V[0, j] + D*V[1, j])
                kick(s, 0.5*h)
                if half == 0:
                    s[0] += h*s[2]; s[1] += h*s[3]
                    for j in range(V.shape[1]):
                        V[0, j] += h*V[2, j]; V[1, j] += h*V[3, j]


@nb.njit(cache=True)
def sali_run(s0, nblk):
    s = s0.copy(); V = np.zeros((4, 2)); V[0, 0] = 1.0; V[2, 1] = 1.0; val = 1.0
    for b in range(nblk):
        tangent(s, V, TAU)
        a = V[:, 0]/math.sqrt(np.sum(V[:, 0]**2)); c = V[:, 1]/math.sqrt(np.sum(V[:, 1]**2))
        val = min(math.sqrt(np.sum((a + c)**2)), math.sqrt(np.sum((a - c)**2)))
        V[:, 0] = a; V[:, 1] = c
        if val < 1e-16:
            break
    return val


@nb.njit(parallel=True, cache=True)
def sali_many(S0, nblk):
    out = np.empty(S0.shape[0])
    for i in nb.prange(S0.shape[0]):
        out[i] = sali_run(S0[i], nblk)
    return out


@nb.njit(cache=True)
def proj(s, v):
    x, y, px, py = s[0], s[1], s[2], s[3]
    g = np.array([x + 2*x*y, y + x*x - y*y, px, py]); g /= math.sqrt(np.sum(g*g))
    f = np.array([px, py, -(x + 2*x*y), -(y + x*x - y*y)]); f -= np.dot(f, g)*g; f /= math.sqrt(np.sum(f*f))
    return v - np.dot(v, g)*g - np.dot(v, f)*f


@nb.njit(cache=True, inline='always')
def wedge(a, b):
    na2 = np.sum(a*a)
    bp = b - (np.dot(a, b)/na2)*a
    return math.sqrt(na2*np.sum(bp*bp))


@nb.njit(parallel=True, cache=True)
def grid_maps(S, ok, hy, hp, nblk):
    """S (N, N, 4) grid states (NaN where inadmissible). returns sup-maps (N, N, 4): S_h, B_h, OFLI2_h, LEON_h."""
    n1, n2 = ok.shape; F = np.full((n1, n2, 4), -np.inf)
    for b in range(nblk):
        for i in nb.prange(n1):                                         # advance all nodes one block
            for j in range(n2):
                if ok[i, j]:
                    orbit(S[i, j], TAU)
        for i in nb.prange(n1):                                         # read out the stencils
            for j in range(n2):
                if not ok[i, j]:
                    continue
                x0 = S[i, j]
                for d in range(2):
                    h = hy if d == 0 else hp
                    if d == 0:
                        if i < 1 or i > n1 - 2 or not (ok[i-1, j] and ok[i+1, j]):
                            continue
                        xp1, xm1 = S[i+1, j], S[i-1, j]
                        has2 = i >= 2 and i <= n1 - 3 and ok[i-2, j] and ok[i+2, j]
                        xp2 = S[i+2, j] if has2 else xp1; xm2 = S[i-2, j] if has2 else xm1
                    else:
                        if j < 1 or j > n2 - 2 or not (ok[i, j-1] and ok[i, j+1]):
                            continue
                        xp1, xm1 = S[i, j+1], S[i, j-1]
                        has2 = j >= 2 and j <= n2 - 3 and ok[i, j-2] and ok[i, j+2]
                        xp2 = S[i, j+2] if has2 else xp1; xm2 = S[i, j-2] if has2 else xm1
                    D = (xp1 - xm1)/(2*h); C = (xp1 + xm1 - 2*x0)/(h*h)
                    pD = proj(x0, D); pC = proj(x0, C); nD = math.sqrt(np.sum(pD*pD))
                    F[i, j, 0] = max(F[i, j, 0], math.log(nD))
                    F[i, j, 1] = max(F[i, j, 1], math.log(max(h*wedge(pD, pC)/(nD*nD), 1e-300)))
                    o = proj(x0, D + 0.5*C); F[i, j, 2] = max(F[i, j, 2], math.log(math.sqrt(np.sum(o*o))))
                    if has2:                                            # quartic collocation -> LEON_h
                        c1 = proj(x0, (8*(xp1 - xm1) - (xp2 - xm2))/12)
                        c4 = proj(x0, (xp2 - 4*xp1 + 6*x0 - 4*xm1 + xm2)/24)
                        n1c = np.sum(c1*c1)
                        F[i, j, 3] = max(F[i, j, 3], math.log(max(wedge(c1, c4)/n1c, 1e-300))/3)
    return F


def auc(y, z):
    o = np.argsort(z, kind='mergesort'); r = np.empty(len(z)); zs = z[o]; i = 0
    while i < len(z):
        j = i
        while j + 1 < len(z) and zs[j+1] == zs[i]:
            j += 1
        r[o[i:j+1]] = 0.5*(i + j + 2); i = j + 1
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1*(n1 + 1)/2)/(n1*n0)


if __name__ == '__main__':
    yv = np.linspace(-0.6, 0.8, N); pv = np.linspace(-0.6, 0.6, N); hy, hp = yv[1] - yv[0], pv[1] - pv[0]
    Y, P = np.meshgrid(yv, pv, indexing='ij'); v = 2*E - P**2 - Y**2 + 2*Y**3/3
    PX = np.sqrt(np.maximum(v, 0)); ok = (v > 0) & (PX >= 1e-3)
    S = np.full((N, N, 4), np.nan); S[..., 1] = Y; S[..., 2] = PX; S[..., 3] = P; S[..., 0] = 0.0; S[~ok] = np.nan
    idx = np.argwhere(ok); S0 = S[ok]; nblk = int(round(T/(TAU*DT)))
    print('admissible nodes', ok.sum(), flush=True)
    # compile on a tiny problem, then time with NTHREADS threads
    sali_many(S0[:4].copy(), 1); grid_maps(S[:8, :8].copy(), ok[:8, :8].copy(), hy, hp, 1)
    nb.set_num_threads(NTHREADS)
    t0 = time.time(); sali_T = sali_many(S0, nblk); t_sali = time.time() - t0
    t0 = time.time(); F = grid_maps(S.copy(), ok, hy, hp, nblk); t_grid = time.time() - t0
    print(f'SALI map {t_sali:.1f} s, grid maps {t_grid:.1f} s ({NTHREADS} threads)', flush=True)
    nb.set_num_threads(nb.config.NUMBA_DEFAULT_NUM_THREADS)
    sub = np.arange(0, len(S0), 7); ref = sali_many(S0[sub], int(round(TREF/(TAU*DT))))
    lab = np.where(ref < 1e-8, 1, np.where(ref > 1e-4, 0, -1)); k = lab >= 0
    sal = np.full((N, N), np.nan); sal[ok] = sali_T
    maps = {'SALI': sal}
    for c, name in enumerate(('S_h', 'B_h', 'OFLI2_h', 'LEON_h')):
        m = F[..., c].copy(); m[~ok | ~np.isfinite(m)] = np.nan; maps[name] = m
    ii, jj = idx[sub][k, 0], idx[sub][k, 1]; aucs = {}
    for name, m in maps.items():
        z = -np.log10(np.maximum(m[ii, jj], 1e-16)) if name == 'SALI' else m[ii, jj]
        good = np.isfinite(z); aucs[name] = auc(lab[k][good], z[good])
        print(f'  AUROC {name:8s} {aucs[name]:.3f}  (n={good.sum()})')
    np.savez(os.path.join(os.path.dirname(__file__), '..', 'data', 'fig4_maps.npz'), y=yv, py=pv, ok=ok,
             ref_ij=idx[sub], ref=ref, t_sali=t_sali, t_grid=t_grid, nthreads=NTHREADS,
             auc_names=np.array(list(aucs)), auc_vals=np.array(list(aucs.values())), n_labelled=int(k.sum()), **maps)
