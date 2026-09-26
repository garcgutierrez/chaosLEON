"""AUROC test of the full-cloud contour indicators (ucci_hh.py) against SALI / OFLI / MEGNO.
Usage:  python ucci_auroc_test.py NPOINTS N0 CAP RETURNS SIGMA OUT.npy SEED
e.g.    python ucci_auroc_test.py 400 16 256 20 1e-3 cheap.npy 21
Labels: SALI(t=1e4) < 1e-8 chaotic, > 1e-4 regular (as in compare_time_auc.py).
Saved per point: (reference SALI, contour wall time, features at returns 2,3,5,10,20,40)
features = (T, stretch_rate, UCCI, UFR, log max_nodes, -log10 SALI(T), OFLI(T), MEGNO(T)).
"""
import sys, io, contextlib, time, numpy as np
sys.path.insert(0, '.')
import ucci_hh as U                                   # keep ucci_hh.py in the same folder
from new_indicators_hh import engineA, sali_ref, px_from, auc, L
rng = np.random.default_rng(int(sys.argv[7])); pts = []
while len(pts) < int(sys.argv[1]):
    y, py = rng.uniform(-.6, .8), rng.uniform(-.6, .6)
    if np.isfinite(px_from(y, py)) and px_from(y, py) > 0.05: pts.append((y, py))
N0, CAP, NRET, SIG, OUT, SEED = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), sys.argv[6], int(sys.argv[7])
cfg = U.Config(returns=NRET, sigma=SIG, probabilities=(0.6,), initial_nodes=N0, max_nodes=CAP).checked()
RET = [r for r in (2, 3, 5, 10, 20, 40) if r <= NRET]
rows = []
for i, (y, py) in enumerate(pts):
    x0 = np.array([0., y, px_from(y, py), py])
    ref = sali_ref(x0, 0.01, 500, 2000)
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            hist, det, _ = U.run_contours(y, py, cfg)
    except Exception as e:
        print('fail', i, e); continue
    wall = time.perf_counter() - t0
    feats = []
    for r in RET:
        h = hist[r-1]; T = h['time']
        hidx = np.array([max(0, int(round(T/5)) - 1)])
        t1 = time.perf_counter(); A = engineA(x0, 0.01, 500, int(hidx[0]) + 1, hidx); w2 = time.perf_counter() - t1
        feats.append((T, h['stretch_rate'], h['ucci'], h['ufr'], np.log(h['max_nodes']), -np.log10(max(A[0, 0], 1e-16)), A[0, 3], A[0, 4]))
    rows.append((ref, wall, feats)); print(i, f'ref={ref:.1e} wall={wall:.1f}s', flush=True)
np.save(OUT, np.array(rows, dtype=object), allow_pickle=True)
