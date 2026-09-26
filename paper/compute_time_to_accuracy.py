"""Data for Sec. VI G and Fig. 11: wall time needed by each chaos map to reach a given accuracy.

Same problem as Fig. 4 and compute_speedup.py (Henon-Heiles, E = 1/8, 600^2 grid, labels = SALI(1e4) on every
seventh admissible node).  For every indicator the AUROC is computed as a function of the integration time T:
  SALI and tangent LEON (y direction) on the labelled nodes, up to T = 2000 and T = 1000;
  grid maps S_h, B_h, OFLI2_h and LEON_h on the full grid (they need the neighbours), up to T = 1000.
Wall times of the full maps as a function of T are measured with p threads (default: all hardware threads) for SALI (whose cost is not linear in T
because of the early exit once SALI < 1e-16); the grid and tangent LEON maps do a fixed amount of work per time step,
so their wall time is t(200) T/200 with t(200) from data/speedup.json.  Single-thread times use the measured speed-up
at T = 200.
Usage:  python compute_time_to_accuracy.py [--threads p] [--out FILE]
        -> ../data/time_to_accuracy.json   (about 15 min with 12 threads on a 6-core i5-10400)
The AUROC values do not depend on the machine; they are recomputed on every machine as a reproducibility check.
The wall times are stored under 'wall' with the thread count in 'wall_threads' ('wall12' in the original file).
"""
import numpy as np, numba as nb, math, time, os, json, argparse
import machine_info
from compute_fig4_maps import E, DT, TAU, orbit, tangent, proj, wedge, auc
from compute_speedup import jet_step, readout, neighbours, ic_jets, sali_map, LOG24

HERE = os.path.dirname(os.path.abspath(__file__))
N = 600
TH = np.array([10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100, 120, 150, 200, 250, 300, 400, 500, 600, 700, 800, 1000,
               1250, 1500, 2000], float)
TGRID_MAX, TLEON_MAX, TSALI_MAX = 1000.0, 1000.0, 2000.0
TWALL = [50, 100, 200, 300, 500, 700, 1000, 1500, 2000]


@nb.njit(parallel=True, cache=True)
def grid_h(S, NB, hy, hp, hb):
    n = S.shape[0]; F = np.full((n, 4), -np.inf); out = np.full((n, hb.size, 4), np.nan); k = 0
    for b in range(hb[-1] + 1):
        for i in nb.prange(n):
            orbit(S[i], TAU)
        for i in nb.prange(n):
            readout(S, NB, i, hy, 0, F); readout(S, NB, i, hp, 1, F)
        if b == hb[k]:
            out[:, k, :] = F; k += 1
    return out


@nb.njit(cache=True)
def sali_run_h(s0, hb):
    s = s0.copy(); V = np.zeros((4, 2)); V[0, 0] = 1.0; V[2, 1] = 1.0; out = np.empty(hb.size); k = 0; val = 1.0
    for b in range(hb[-1] + 1):
        if val >= 1e-16:
            tangent(s, V, TAU)
            a = V[:, 0]/math.sqrt(np.sum(V[:, 0]**2)); c = V[:, 1]/math.sqrt(np.sum(V[:, 1]**2))
            val = min(math.sqrt(np.sum((a + c)**2)), math.sqrt(np.sum((a - c)**2))); V[:, 0] = a; V[:, 1] = c
        if b == hb[k]:
            out[k] = val; k += 1
    return out


@nb.njit(parallel=True, cache=True)
def sali_h(S0, hb):
    out = np.empty((S0.shape[0], hb.size))
    for i in nb.prange(S0.shape[0]):
        out[i] = sali_run_h(S0[i], hb)
    return out


@nb.njit(cache=True)
def leon_run_h(x0, Z0, hb):
    s = x0.copy(); Z = Z0.copy(); Lc = 0.0; sup = -1e300; out = np.empty(hb.size); k = 0
    for b in range(hb[-1] + 1):
        jet_step(s, Z, TAU)
        p0 = proj(s, Z[0]); p3 = proj(s, Z[3]); n1 = math.sqrt(np.sum(p0*p0))
        sup = max(sup, Lc + (math.log(max(wedge(p0, p3), 1e-300)) - 2*math.log(n1) - LOG24)/3)
        c = math.sqrt(np.sum(Z[0]*Z[0])); Lc += math.log(c)
        for j in range(4):
            Z[j] /= c**(j + 1)
        if b == hb[k]:
            out[k] = sup; k += 1
    return out


@nb.njit(parallel=True, cache=True)
def leon_h(S0, Z0, hb):
    out = np.empty((S0.shape[0], hb.size))
    for i in nb.prange(S0.shape[0]):
        out[i] = leon_run_h(S0[i], Z0[i], hb)
    return out


def blocks(Tmax):
    T = TH[TH <= Tmax]; return T, (np.round(T/(TAU*DT)).astype(np.int64) - 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=None, help='threads (default: all hardware threads)')
    ap.add_argument('--out', default=os.path.join(HERE, '..', 'data', 'time_to_accuracy.json'))
    a = ap.parse_args(argv); MI = machine_info.info()
    p = min(a.threads or MI['n_logical'], nb.config.NUMBA_NUM_THREADS)
    yv = np.linspace(-0.6, 0.8, N); pv = np.linspace(-0.6, 0.6, N); hy, hp = yv[1] - yv[0], pv[1] - pv[0]
    Y, P = np.meshgrid(yv, pv, indexing='ij'); v = 2*E - P**2 - Y**2 + 2*Y**3/3
    PX = np.sqrt(np.maximum(v, 0)); ok = (v > 0) & (PX >= 1e-3)
    Sg = np.full((N, N, 4), np.nan); Sg[..., 0] = 0.0; Sg[..., 1] = Y; Sg[..., 2] = PX; Sg[..., 3] = P
    idx, NB = neighbours(ok); S0 = Sg[ok].copy(); n = len(S0)
    ref = np.load(os.path.join(HERE, '..', 'data', 'fig4_maps.npz'))
    sub = np.arange(0, n, 7); assert np.array_equal(idx[sub], ref['ref_ij'])
    lab = np.where(ref['ref'] < 1e-8, 1, np.where(ref['ref'] > 1e-4, 0, -1)); kk = lab >= 0
    Ssub = np.ascontiguousarray(S0[sub]); Zsub = ic_jets(Ssub[:, 1], Ssub[:, 3])
    nb.set_num_threads(p); res = dict(T={}, auroc={}, wall={}, wall_threads=p, n_labelled=int(kk.sum()), machine=MI)
    print(f'{p} threads on', MI, flush=True)

    def score(name, T, Z, neg_log=False):
        a = []
        for j in range(Z.shape[1]):
            z = Z[:, j]
            z = -np.log10(np.maximum(z, 1e-16)) if neg_log else z
            g = np.isfinite(z); a.append(float(auc(lab[kk][g[kk]], z[kk][g[kk]])))
        res['T'][name] = T.tolist(); res['auroc'][name] = a; print(name, dict(zip(T.tolist(), np.round(a, 3))), flush=True)

    T, hb = blocks(TSALI_MAX); t0 = time.time(); score('SALI', T, sali_h(Ssub, hb), True); print(f'  {time.time()-t0:.0f} s')
    T, hb = blocks(TLEON_MAX); t0 = time.time(); score('LEON', T, leon_h(Ssub, Zsub, hb)); print(f'  {time.time()-t0:.0f} s')
    T, hb = blocks(TGRID_MAX); t0 = time.time(); G = grid_h(S0.copy(), NB, hy, hp, hb)[sub]
    print(f'  grid {time.time()-t0:.0f} s')
    for c, name in enumerate(('S_h', 'B_h', 'OFLI2_h', 'LEON_h')):
        Z = G[:, :, c]; Z[~np.isfinite(Z)] = np.nan; score(name, T, Z)

    # wall time of the full SALI map versus T (p threads)
    for Tw in TWALL:
        nblk = int(round(Tw/(TAU*DT))); t0 = time.perf_counter(); sali_map(S0, nblk); res['wall'][Tw] = time.perf_counter() - t0
        print(f'SALI map T={Tw}: {res["wall"][Tw]:.1f} s ({p} threads)', flush=True)
        with open(a.out, 'w') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
