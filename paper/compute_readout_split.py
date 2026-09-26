"""Cost of each grid indicator on its own (supplement to Sec. VI G and Fig. 10).

The four grid maps S_h, B_h, OFLI2_h and LEON_h are read out from the same orbits, so compute_speedup.py times them
together ('grid') and the orbit integration alone ('grid_adv').  Here the read-out computes one indicator at a time:
  grid_all   all four (as 'grid' in compute_speedup.py)
  grid_adv   advance phase only (no read-out): the lower limit for any grid indicator
  S_h, B_h, OFLI2_h   five-point stencil;  LEON_h   nine-point stencil
Same problem as compute_speedup.py (Henon-Heiles, E = 1/8, 600^2 grid, T = 200), timed with 1 thread, one thread per
physical core and all hardware threads.  Each single-indicator map is checked against data/fig4_maps.npz.
Usage:  python compute_readout_split.py [--threads 1,6,12] [--out FILE]   -> ../data/readout_split.json
        (run_benchmarks.py calls it and writes ../data/machines/<host>/readout_split.json)
"""
import numpy as np, numba as nb, math, time, os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from compute_fig4_maps import E, T, DT, TAU, orbit, proj, wedge
from compute_speedup import neighbours
import machine_info

NAMES = ('S_h', 'B_h', 'OFLI2_h', 'LEON_h')


@nb.njit(cache=True, inline='always')
def readout_one(S, NB, k, h, d, F, mode):
    """Read-out of indicator `mode` only (0 S_h, 1 B_h, 2 OFLI2_h, 3 LEON_h), same formulas as compute_speedup.readout."""
    x0 = S[k]; i1, j1, i2, j2 = NB[k, d, 0], NB[k, d, 1], NB[k, d, 2], NB[k, d, 3]
    if i1 < 0 or j1 < 0:
        return
    xp1 = S[i1]; xm1 = S[j1]
    if mode == 0:
        pD = proj(x0, (xp1 - xm1)/(2*h)); F[k, 0] = max(F[k, 0], math.log(math.sqrt(np.sum(pD*pD))))
    elif mode == 1:
        pD = proj(x0, (xp1 - xm1)/(2*h)); pC = proj(x0, (xp1 + xm1 - 2*x0)/(h*h)); nD = math.sqrt(np.sum(pD*pD))
        F[k, 1] = max(F[k, 1], math.log(max(h*wedge(pD, pC)/(nD*nD), 1e-300)))
    elif mode == 2:
        o = proj(x0, (xp1 - xm1)/(2*h) + 0.5*(xp1 + xm1 - 2*x0)/(h*h))
        F[k, 2] = max(F[k, 2], math.log(math.sqrt(np.sum(o*o))))
    elif i2 >= 0 and j2 >= 0:
        xp2 = S[i2]; xm2 = S[j2]
        c1 = proj(x0, (8*(xp1 - xm1) - (xp2 - xm2))/12)
        c4 = proj(x0, (xp2 - 4*xp1 + 6*x0 - 4*xm1 + xm2)/24)
        F[k, 3] = max(F[k, 3], math.log(max(wedge(c1, c4)/np.sum(c1*c1), 1e-300))/3)


@nb.njit(parallel=True, cache=True)
def grid_one(S, NB, hy, hp, nblk, mode):
    n = S.shape[0]; F = np.full((n, 4), -np.inf)
    for b in range(nblk):
        for k in nb.prange(n):
            orbit(S[k], TAU)
        for k in nb.prange(n):
            readout_one(S, NB, k, hy, 0, F, mode); readout_one(S, NB, k, hp, 1, F, mode)
    return F


def main(argv=None):
    from compute_speedup import grid_flat
    ap = argparse.ArgumentParser(); ap.add_argument('--threads'); ap.add_argument('--out'); ap.add_argument('--N', type=int, default=600)
    a = ap.parse_args(argv); MI = machine_info.info(); nc, nl = MI['n_cores'], MI['n_logical']
    P = [int(p) for p in a.threads.split(',')] if a.threads else sorted({1, nc, nl})
    P = [p for p in P if p <= nb.config.NUMBA_NUM_THREADS]
    out = a.out or os.path.join(HERE, '..', 'data', 'readout_split.json')
    N = a.N
    yv = np.linspace(-0.6, 0.8, N); pv = np.linspace(-0.6, 0.6, N); hy, hp = yv[1] - yv[0], pv[1] - pv[0]
    Y, P2 = np.meshgrid(yv, pv, indexing='ij'); v = 2*E - P2**2 - Y**2 + 2*Y**3/3
    PX = np.sqrt(np.maximum(v, 0)); ok = (v > 0) & (PX >= 1e-3)
    Sg = np.zeros((N, N, 4)); Sg[..., 1] = Y; Sg[..., 2] = PX; Sg[..., 3] = P2
    idx, NB = neighbours(ok); S0 = Sg[ok].copy(); n = len(S0); nblk = int(round(T/(TAU*DT)))
    print(f'N={N}, {n} nodes, threads {P}', flush=True)
    for m in range(4):                                             # compile
        grid_one(S0[:8].copy(), -np.ones((8, 2, 4), dtype=np.int64), hy, hp, 1, m)
    grid_flat(S0[:8].copy(), -np.ones((8, 2, 4), dtype=np.int64), hy, hp, 1, True)
    res = dict(N=N, n_nodes=int(n), T=T, threads=P, machine=MI, times={k: {} for k in ('grid_all', 'grid_adv') + NAMES},
               check={})
    ref = np.load(os.path.join(HERE, '..', 'data', 'fig4_maps.npz'))

    def tm(f, *x):
        t0 = time.perf_counter(); r = f(*x); return time.perf_counter() - t0, r

    for p in P:
        nb.set_num_threads(p); W = res['times']
        W['grid_all'][p], _ = tm(grid_flat, S0.copy(), NB, hy, hp, nblk, True)
        W['grid_adv'][p], _ = tm(grid_flat, S0.copy(), NB, hy, hp, nblk, False)
        for m, name in enumerate(NAMES):
            W[name][p], F = tm(grid_one, S0.copy(), NB, hy, hp, nblk, m)
            if p == P[0] and N == 600:
                g = np.full((N, N), np.nan); g[ok] = F[:, m]; g[~np.isfinite(g)] = np.nan; r = ref[name]
                both = np.isfinite(g) & np.isfinite(r)
                res['check'][name + '_maxdiff'] = float(np.max(np.abs(g[both] - r[both])))
        print(f'p={p:2d}: ' + ', '.join(f'{k} {W[k][p]:.1f} s' for k in W), flush=True)
        with open(out, 'w') as f:
            json.dump(res, f, indent=1)
    print('checks', res['check'], flush=True)
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
