"""Data for Sec. VI G and Fig. 11: preliminary strong-scaling (parallel speed-up) study.

Fixed problem, as in Fig. 4: Henon-Heiles, E = 1/8, section x = 0, N^2 grid in (y, p_y) (N = 600: 180 594 admissible
nodes), T = 200, Yoshida-4 with dt = 0.01, read-out/renormalization every TAU = 500 steps (5 time units).
Workloads, each timed with p = 1, 2, 3, 4, 6, 8, 10, 12 numba threads:
  sali       SALI map: orbit + two deviation vectors per node (independent orbits, early exit once SALI < 1e-16;
             nodes are processed in a random order to balance the early exits)
  leon       tangent LEON map: orbit + first to fourth variations along the y direction (independent orbits)
  grid       grid-as-ensemble maps S_h, B_h, OFLI2_h and LEON_h from the orbits of the nodes alone.  Block-synchronous:
             advance all admissible nodes by TAU steps (parallel), barrier, read out the nine-point stencils (parallel)
  grid_adv   the advance phase of `grid` alone (the read-out costs grid - grid_adv)
  grid_rows  the row-parallel implementation of compute_fig4_maps.py (p = 1, 6, 12 only): rows near the edge of the
             admissible region contain few nodes, so a static partition of the rows is unbalanced
The grid maps of the flat implementation are checked against data/fig4_maps.npz, and the AUROC of the tangent LEON map
is computed on the labelled nodes of Fig. 4.
Usage:  python compute_speedup.py [N] [--threads 1,2,4,...] [--out FILE]
        default: thread counts from machine_info.default_threads (1-4, fractions of the physical cores, up to all
        hardware threads; on the 6-core i5-10400: 1, 2, 3, 4, 6, 8, 10, 12), output ../data/speedup.json
        (about 45 min on a 6-core i5-10400).  Use run_benchmarks.py to benchmark a new machine.
"""
import numpy as np, numba as nb, math, time, os, sys, json, platform, argparse
from compute_fig4_maps import (E, T, DT, TAU, W1, W0, kick, orbit, sali_run, proj, wedge, grid_maps, auc)
import machine_info

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'data', 'speedup.json')
LOG24 = math.log(24.0)


# ----------------------------------------------------------------------------- independent-orbit maps
@nb.njit(parallel=True, cache=True)
def sali_map(S0, nblk):
    out = np.empty(S0.shape[0])
    for k in nb.prange(S0.shape[0]):
        out[k] = sali_run(S0[k], nblk)
    return out


@nb.njit(cache=True, inline='always')
def hh_df(x, y, a):
    return (-(1 + 2*y)*a[0] - 2*x*a[1], -2*x*a[0] - (1 - 2*y)*a[1])


@nb.njit(cache=True, inline='always')
def hh_d2(a, b):
    return (-2*(a[0]*b[1] + a[1]*b[0]), -2*a[0]*b[0] + 2*a[1]*b[1])


@nb.njit(cache=True)
def jet_step(s, Z, n):
    """Orbit and exact discrete first..fourth variations Z[0..3] of the Yoshida-4 leapfrog (cubic potential)."""
    for _ in range(n):
        for c in (W1, W0, W1):
            h = c*DT
            for half in range(2):
                hk = 0.5*h; x, y = s[0], s[1]
                a = hh_df(x, y, Z[0])
                b = hh_df(x, y, Z[1]); b2 = hh_d2(Z[0], Z[0])
                g = hh_df(x, y, Z[2]); g2 = hh_d2(Z[0], Z[1])
                e = hh_df(x, y, Z[3]); e2 = hh_d2(Z[0], Z[2]); e3 = hh_d2(Z[1], Z[1])
                for i in range(2):
                    Z[0, 2+i] += hk*a[i]; Z[1, 2+i] += hk*(b[i] + b2[i]); Z[2, 2+i] += hk*(g[i] + 3*g2[i])
                    Z[3, 2+i] += hk*(e[i] + 4*e2[i] + 3*e3[i])
                kick(s, hk)
                if half == 0:
                    s[0] += h*s[2]; s[1] += h*s[3]
                    for k in range(4):
                        Z[k, 0] += h*Z[k, 2]; Z[k, 1] += h*Z[k, 3]


@nb.njit(cache=True)
def leon_run(x0, Z0, nblk):
    s = x0.copy(); Z = Z0.copy(); Lc = 0.0; sup = -1e300
    for b in range(nblk):
        jet_step(s, Z, TAU)
        p0 = proj(s, Z[0]); p3 = proj(s, Z[3]); n1 = math.sqrt(np.sum(p0*p0))
        sup = max(sup, Lc + (math.log(max(wedge(p0, p3), 1e-300)) - 2*math.log(n1) - LOG24)/3)
        c = math.sqrt(np.sum(Z[0]*Z[0])); Lc += math.log(c)
        for k in range(4):
            Z[k] /= c**(k + 1)
    return sup


@nb.njit(parallel=True, cache=True)
def leon_map(S0, Z0, nblk):
    out = np.empty(S0.shape[0])
    for k in nb.prange(S0.shape[0]):
        out[k] = leon_run(S0[k], Z0[k], nblk)
    return out


def ic_jets(Y, P):
    """Z_k(0), k = 1..4, of the iso-energetic IC curve (0, y + xi, p_x(y + xi, p_y), p_y)."""
    n = Y.size; g = np.zeros((5, n))
    g[0] = 2*E - P*P - Y*Y + 2*Y**3/3; g[1] = -2*Y + 2*Y*Y; g[2] = -1 + 2*Y; g[3] = 2/3
    u = np.zeros((5, n)); u[0] = np.sqrt(g[0])
    for k in range(1, 5):
        u[k] = (g[k] - sum(u[j]*u[k-j] for j in range(1, k)))/(2*u[0])
    Z = np.zeros((n, 4, 4)); Z[:, 0, 1] = 1.0
    for k in range(1, 5):
        Z[:, k-1, 2] = math.factorial(k)*u[k]
    return Z


# ----------------------------------------------------------------------------- grid-as-ensemble, flat node list
@nb.njit(cache=True, inline='always')
def readout(S, NB, k, h, d, F):
    x0 = S[k]; i1, j1, i2, j2 = NB[k, d, 0], NB[k, d, 1], NB[k, d, 2], NB[k, d, 3]
    if i1 < 0 or j1 < 0:
        return
    xp1 = S[i1]; xm1 = S[j1]; has2 = i2 >= 0 and j2 >= 0
    xp2 = S[i2] if has2 else xp1; xm2 = S[j2] if has2 else xm1
    D = (xp1 - xm1)/(2*h); C = (xp1 + xm1 - 2*x0)/(h*h)
    pD = proj(x0, D); pC = proj(x0, C); nD = math.sqrt(np.sum(pD*pD))
    F[k, 0] = max(F[k, 0], math.log(nD))
    F[k, 1] = max(F[k, 1], math.log(max(h*wedge(pD, pC)/(nD*nD), 1e-300)))
    o = proj(x0, D + 0.5*C); F[k, 2] = max(F[k, 2], math.log(math.sqrt(np.sum(o*o))))
    if has2:
        c1 = proj(x0, (8*(xp1 - xm1) - (xp2 - xm2))/12)
        c4 = proj(x0, (xp2 - 4*xp1 + 6*x0 - 4*xm1 + xm2)/24)
        F[k, 3] = max(F[k, 3], math.log(max(wedge(c1, c4)/np.sum(c1*c1), 1e-300))/3)


@nb.njit(parallel=True, cache=True)
def grid_flat(S, NB, hy, hp, nblk, do_readout):
    """S (n, 4) admissible states; NB (n, 2, 4) flat indices of the neighbours +1, -1, +2, -2 (or -1)."""
    n = S.shape[0]; F = np.full((n, 4), -np.inf)
    for b in range(nblk):
        for k in nb.prange(n):                    # advance phase: independent orbits
            orbit(S[k], TAU)
        if do_readout:                            # (implicit barrier) read-out phase: stencils
            for k in nb.prange(n):
                readout(S, NB, k, hy, 0, F); readout(S, NB, k, hp, 1, F)
    return F


def neighbours(ok):
    N1, N2 = ok.shape; K = -np.ones(ok.shape, dtype=np.int64); idx = np.argwhere(ok); K[ok] = np.arange(len(idx))
    Kp = np.pad(K, 2, constant_values=-1); NB = np.empty((len(idx), 2, 4), dtype=np.int64)
    i, j = idx[:, 0] + 2, idx[:, 1] + 2
    for d, (di, dj) in enumerate(((1, 0), (0, 1))):
        for m, s in enumerate((1, -1, 2, -2)):
            NB[:, d, m] = Kp[i + s*di, j + s*dj]
    return idx, NB


# ----------------------------------------------------------------------------- driver
def timed(f, *a):
    t0 = time.perf_counter(); r = f(*a); return time.perf_counter() - t0, r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('N', nargs='?', type=int, default=600)
    ap.add_argument('--threads', default=None, help='comma-separated thread counts (default: automatic)')
    ap.add_argument('--out', default=None, help='output json (default ../data/speedup.json, _N<N> if N != 600)')
    a = ap.parse_args(argv); N = a.N
    MI = machine_info.info(); nc, nl = MI['n_cores'], MI['n_logical']
    PLIST = [int(p) for p in a.threads.split(',')] if a.threads else machine_info.default_threads(nc, nl)
    PLIST = [p for p in PLIST if p <= nb.config.NUMBA_NUM_THREADS]
    PSPECIAL = sorted({1, min(nc, PLIST[-1]), PLIST[-1]})           # thread counts of the extra variants and repeats
    out_file = a.out or (OUT if N == 600 else OUT.replace('.json', f'_N{N}.json'))
    yv = np.linspace(-0.6, 0.8, N); pv = np.linspace(-0.6, 0.6, N); hy, hp = yv[1] - yv[0], pv[1] - pv[0]
    Y, P = np.meshgrid(yv, pv, indexing='ij'); v = 2*E - P**2 - Y**2 + 2*Y**3/3
    PX = np.sqrt(np.maximum(v, 0)); ok = (v > 0) & (PX >= 1e-3)
    Sg = np.full((N, N, 4), np.nan); Sg[..., 0] = 0.0; Sg[..., 1] = Y; Sg[..., 2] = PX; Sg[..., 3] = P; Sg[~ok] = np.nan
    idx, NB = neighbours(ok); S0 = Sg[ok].copy(); n = len(S0); nblk = int(round(T/(TAU*DT)))
    Z0 = ic_jets(S0[:, 1], S0[:, 3])
    perm = np.random.default_rng(1).permutation(n)
    S0p = np.ascontiguousarray(S0[perm]); Z0p = np.ascontiguousarray(Z0[perm])
    print(f'N={N}, admissible nodes {n}, {nblk} blocks, threads {PLIST}', flush=True)
    print('machine', MI, flush=True)

    # compile on small problems
    sali_map(S0[:8].copy(), 1); leon_map(S0[:8].copy(), Z0[:8].copy(), 1)
    grid_flat(S0[:8].copy(), -np.ones((8, 2, 4), dtype=np.int64), hy, hp, 1, True)
    grid_maps(Sg[:8, :8].copy(), ok[:8, :8].copy(), hy, hp, 1)
    try:
        layer = nb.threading_layer()
    except Exception:
        layer = 'unknown'
    res = dict(N=N, n_nodes=int(n), T=T, dt=DT, tau_steps=TAU, n_blocks=nblk, threads=PLIST,
               cpu=platform.processor(), os=platform.platform(), numba=nb.__version__, threading_layer=layer,
               max_threads=nb.config.NUMBA_DEFAULT_NUM_THREADS, machine=MI, times={}, repeat={}, check={})
    W = res['times']
    for key in ('sali', 'leon', 'grid', 'grid_adv', 'grid_rows', 'sali_natural'):
        W[key] = {}

    def save():
        with open(out_file, 'w') as f:
            json.dump(res, f, indent=1)

    for p in PLIST:
        nb.set_num_threads(p)
        W['sali'][p], _ = timed(sali_map, S0p, nblk)
        W['leon'][p], Lp = timed(leon_map, S0p, Z0p, nblk)
        W['grid'][p], F = timed(grid_flat, S0.copy(), NB, hy, hp, nblk, True)
        W['grid_adv'][p], _ = timed(grid_flat, S0.copy(), NB, hy, hp, nblk, False)
        if p in PSPECIAL:
            W['grid_rows'][p], _ = timed(grid_maps, Sg.copy(), ok, hy, hp, nblk)
            W['sali_natural'][p], _ = timed(sali_map, S0, nblk)
        print(f'p={p:2d}: ' + ', '.join(f'{k} {W[k][p]:.1f} s' for k in W if p in W[k]), flush=True)
        save()

    # run-to-run variability
    for p in (1, PLIST[-1]):
        nb.set_num_threads(p)
        res['repeat'][p] = dict(sali=timed(sali_map, S0p, nblk)[0], leon=timed(leon_map, S0p, Z0p, nblk)[0],
                                grid=timed(grid_flat, S0.copy(), NB, hy, hp, nblk, True)[0])
        print(f'repeat p={p}: {res["repeat"][p]}', flush=True)
        save()

    # checks: flat grid maps against Fig. 4, AUROC of the tangent LEON map on the labelled nodes of Fig. 4
    if N != 600:
        return
    ref = np.load(os.path.join(HERE, '..', 'data', 'fig4_maps.npz'))
    for c, name in enumerate(('S_h', 'B_h', 'OFLI2_h', 'LEON_h')):
        m = np.full((N, N), np.nan); m[ok] = F[:, c]; m[~np.isfinite(m)] = np.nan; r = ref[name]
        both = np.isfinite(m) & np.isfinite(r)
        res['check'][name + '_maxdiff'] = float(np.max(np.abs(m[both] - r[both])))
        res['check'][name + '_nan_mismatch'] = int(np.sum(np.isfinite(m) != np.isfinite(r)))
    L = np.empty(n); L[perm] = Lp; Lm = np.full((N, N), np.nan); Lm[ok] = L
    lab = np.where(ref['ref'] < 1e-8, 1, np.where(ref['ref'] > 1e-4, 0, -1)); kk = lab >= 0
    ii, jj = ref['ref_ij'][kk, 0], ref['ref_ij'][kk, 1]; z = Lm[ii, jj]; good = np.isfinite(z)
    res['check']['auroc_LEON_tangent_T200'] = float(auc(lab[kk][good], z[good]))
    res['check']['auroc_fig4'] = dict(zip([str(s) for s in ref['auc_names']], map(float, ref['auc_vals'])))
    print('checks', res['check'], flush=True)
    save()


if __name__ == '__main__':
    main()
