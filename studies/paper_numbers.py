import numpy as np, json, time, numba as nb, sys
from aucnew_lib import auc, L
from newind import engineA, engineB, sali_ref, px_from
from fold import fold_all2, fold_run2
from fold3 import fold3_all
from h3 import WW
rng0 = np.random.default_rng(123)
def ci(y, s, nb_=500):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; r = []
    for _ in range(nb_):
        ix = np.r_[rng0.choice(pos, len(pos)), rng0.choice(neg, len(neg))]; r.append(auc(y[ix], s[ix]))
    return [float(x) for x in np.quantile(r, [.025, .975])]
@nb.njit(parallel=True, cache=True)
def eAB(X, hidx, nbk):
    A = np.empty((X.shape[0], hidx.size, 6)); B = np.empty((X.shape[0], hidx.size))
    for i in nb.prange(X.shape[0]):
        A[i] = engineA(X[i], 0.01, 500, nbk, hidx); B[i] = engineB(X[i], 0.01, 500, nbk, hidx)
    return A, B
out = {}
# ---------- Henon-Heiles, grid-neighbour sets ----------
for tag, fn in [('HH_E0.125', 'gnb_1000.npz'), ('HH_E0.1', 'gnb_1000_E0.1.npz')]:
    d = np.load(fn); F, ref, H, S0 = d['F'], d['ref'], d['H'], d['S0']
    lab = np.where(ref < 1e-8, 1, np.where(ref > 1e-4, 0, -1)); k = lab >= 0; y = lab[k]
    hidx = (H/5-1).astype(np.int64); X = S0[:, 0].copy()
    A, B = eAB(X, hidx, int(H[-1]/5)); Fo = fold_all2(X, np.array([1., 0, 0, 0]), 0.01, 500, int(H[-1]/5), hidx)
    sc = {'SALI': L(A[:, :, 0]), 'GALI3': L(B), 'OFLI': A[:, :, 3], 'MEGNO': A[:, :, 4], 'OFLI2': Fo[:, :, 3], 'bend2': Fo[:, :, 1],
          'grid_stretch': F[:, :, 0], 'grid_fold': F[:, :, 2]}
    res = {'n': int(k.sum()), 'chaotic_frac': float(y.mean()), 'h': [float(d['hy']), float(d['hp'])], 'T': H.tolist(), 'auc': {}, 'ci': {}}
    for m, v in sc.items():
        res['auc'][m] = [float(auc(y, v[k, h])) for h in range(len(H))]
        res['ci'][m] = {int(T): ci(y, v[k, h]) for h, T in enumerate(H) if T in (100, 200)}
    out[tag] = res
    np.savez(f'paper_{tag}.npz', **{m: v for m, v in sc.items()}, lab=lab, H=H)
# resolution study (HH, E=1/8)
out['HH_resolution'] = {}
for N in (250, 500, 1000, 2000):
    d = np.load(f'gnb_{N}.npz'); F, ref, H = d['F'], d['ref'], d['H']; lab = np.where(ref < 1e-8, 1, np.where(ref > 1e-4, 0, -1)); k = lab >= 0
    out['HH_resolution'][N] = {'h': float(d['hy']), 'fold': [float(auc(lab[k], F[k, h, 2])) for h in range(len(H))],
                               'stretch': [float(auc(lab[k], F[k, h, 0])) for h in range(len(H))],
                               'align_inv': [float(1-auc(lab[k], L(F[k, h, 1]))) for h in range(len(H))], 'T': H.tolist()}
# ---------- UFR contour sets (398 points) ----------
out['UFR'] = {}
for s, fn in [('1e-4', 'ucci/cheap_s1e-4.npy'), ('1e-3', 'ucci/cheap.npy'), ('1e-2', 'ucci/cheap_s1e-2.npy')]:
    R = np.load(fn, allow_pickle=True); ref = np.array([r[0] for r in R]); wall = np.array([r[1] for r in R])
    lab = np.where(ref < 1e-8, 1, np.where(ref > 1e-4, 0, -1)); k = lab >= 0
    Fz = np.array([[f for f in r[2]] for r in R])[k]; y = lab[k]
    out['UFR'][s] = {'n': int(k.sum()), 'T': Fz[:, :, 0].mean(0).tolist(), 'UFR': [float(auc(y, Fz[:, j, 3])) for j in range(Fz.shape[1])],
                     'stretch': [float(auc(y, Fz[:, j, 1])) for j in range(Fz.shape[1])], 'UCCI': [float(auc(y, Fz[:, j, 2])) for j in range(Fz.shape[1])],
                     'SALI': [float(auc(y, Fz[:, j, 5])) for j in range(Fz.shape[1])], 'OFLI': [float(auc(y, Fz[:, j, 6])) for j in range(Fz.shape[1])],
                     'wall_mean': float(wall.mean()), 'failed': int(400-len(R))}
# second-order on the UFR point set
rng = np.random.default_rng(21); pts = []
while len(pts) < 400:
    yy, py = rng.uniform(-.6, .8), rng.uniform(-.6, .6)
    if np.isfinite(px_from(yy, py)) and px_from(yy, py) > 0.05: pts.append((0., yy, px_from(yy, py), py))
X = np.array(pts); ref = np.array([sali_ref(x, 0.01, 500, 2000) for x in X]); lab = np.where(ref < 1e-8, 1, np.where(ref > 1e-4, 0, -1)); k = lab >= 0
Hs = np.array([13, 20, 33, 66, 131]); Fo = fold_all2(X, np.array([1., 0, 0, 0]), 0.01, 100, 131, (Hs-1).astype(np.int64))
out['UFR']['second_order_same_points'] = {'T': Hs.tolist(), 'OFLI2': [float(auc(lab[k], Fo[k, h, 3])) for h in range(5)], 'bend2': [float(auc(lab[k], Fo[k, h, 1])) for h in range(5)]}
# ---------- 3-dof ----------
out['3DOF'] = {}
for tag, fn in [('E0.06', 'g3_06_1000.npz'), ('E0.07', 'g3_07_1000.npz'), ('E0.09', 'g3_09_1000.npz'), ('E0.07_N250', 'g3_07_250.npz'), ('E0.07_N4000', 'g3_07_4000.npz')]:
    g = np.load(fn); S0, F, A, B, ref, H = g['S0'], g['F'], g['A'], g['B'], g['ref'], g['H']
    lab = np.where(ref < 1e-8, 1, np.where(ref > 1e-4, 0, -1)); k = lab >= 0; y = lab[k]
    U = S0[:, 1] - S0[:, 2]; U /= np.linalg.norm(U, axis=1)[:, None]
    Fo = fold3_all(S0[:, 0].copy(), U, 0.01, 500, int(H[-1]/5), (H/5-1).astype(np.int64), WW)
    sc = {'SALI': L(A[:, :, 0]), 'GALI3': L(B), 'OFLI': A[:, :, 3], 'MEGNO': A[:, :, 4], 'OFLI2': Fo[:, :, 1], 'bend2': Fo[:, :, 0],
          'grid_stretch': F[:, :, 0], 'grid_fold': F[:, :, 2]}
    out['3DOF'][tag] = {'n': int(k.sum()), 'chaotic_frac': float(y.mean()), 'h': [float(g['h1']), float(g['h2'])], 'T': H.tolist(),
                        'auc': {m: [float(auc(y, v[k, h])) for h in range(len(H))] for m, v in sc.items()},
                        'ci': {m: {int(T): ci(y, v[k, h]) for h, T in enumerate(H) if T in (500, 1000)} for m, v in sc.items()}}
# ---------- timings (single thread, per unit time) ----------
from gridnb import y4n
x = np.array([0., 0.1, px_from(0.1, 0.), 0.]); hid = np.array([99])
S1 = x[None].copy(); y4n(S1, 0.01, 10); sali_ref(x, 0.01, 10, 1); fold_run2(x, np.array([1., 0, 0, 0]), 0.01, 10, 2, np.array([0])); engineA(x, 0.01, 10, 2, np.array([0]))
def tm(f, rep=5):
    t0 = time.perf_counter(); [f() for _ in range(rep)]; return (time.perf_counter()-t0)/rep/500.0
out['cost_per_time_unit'] = {'orbit_only': tm(lambda: y4n(x[None].copy(), 0.01, 50000)), 'SALI': tm(lambda: sali_ref(x, 0.01, 500, 100)),
                             'OFLI_MEGNO_SALI_engine': tm(lambda: engineA(x, 0.01, 500, 100, hid)), 'second_order': tm(lambda: fold_run2(x, np.array([1., 0, 0, 0]), 0.01, 500, 100, hid)),
                             'UFR_contour_16_256': out['UFR']['1e-3']['wall_mean']/131.0}
fg = np.load('fullgrid_600_T200.npz'); out['fullgrid_600'] = {'t_fd': float(fg['t_fd']), 't_sali': float(fg['t_sali']), 'nodes': int(fg['valid'].sum())}
json.dump(out, open('paper/numbers.json', 'w'), indent=1)
print(json.dumps({k: (v if k in ('cost_per_time_unit', 'fullgrid_600') else '...') for k, v in out.items()}, indent=1))
