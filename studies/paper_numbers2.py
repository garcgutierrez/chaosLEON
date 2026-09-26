import numpy as np, json
from aucnew_lib import auc, L
rng0 = np.random.default_rng(321)
def ci(y, s, nb_=500):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; r = []
    for _ in range(nb_):
        ix = np.r_[rng0.choice(pos, len(pos)), rng0.choice(neg, len(neg))]; r.append(auc(y[ix], s[ix]))
    return [float(x) for x in np.quantile(r, [.025, .975])]
D = json.load(open('paper/numbers.json'))
for tag, fn in [('HH_E0.125', 'gnb2_1000.npz'), ('HH_E0.1', 'gnb2_1000_E0.1.npz')]:
    d = np.load(fn); F, ref, H = d['F'], d['ref'], d['H']; lab = np.where(ref < 1e-8, 1, np.where(ref > 1e-4, 0, -1)); k = lab >= 0; y = lab[k]
    assert np.all(np.load(f'paper_{tag}.npz')['lab'] == lab)
    D[tag]['auc']['grid_ofli2'] = [float(auc(y, F[k, h, 3])) for h in range(len(H))]
    D[tag]['auc']['grid_ofli2_fs'] = [float(auc(y, F[k, h, 4])) for h in range(len(H))]
    D[tag]['ci']['grid_ofli2'] = {int(T): ci(y, F[k, h, 3]) for h, T in enumerate(H) if T in (100, 200)}
for N, fn in [(1000, 'gnb2_1000.npz'), (2000, 'gnb2_2000.npz')]:
    d = np.load(fn); F, ref = d['F'], d['ref']; lab = np.where(ref < 1e-8, 1, np.where(ref > 1e-4, 0, -1)); k = lab >= 0
    D['HH_resolution'][str(N)]['grid_ofli2'] = [float(auc(lab[k], F[k, h, 3])) for h in range(F.shape[1])]
for tag, fn in [('E0.06', 'g3b_06.npz'), ('E0.07', 'g3b_07.npz'), ('E0.09', 'g3b_09.npz')]:
    g = np.load(fn); F, ref, H = g['F'], g['ref'], g['H']; lab = np.where(ref < 1e-8, 1, np.where(ref > 1e-4, 0, -1)); k = lab >= 0; y = lab[k]
    D['3DOF'][tag]['auc']['grid_ofli2'] = [float(auc(y, F[k, h, 3])) for h in range(len(H))]
    D['3DOF'][tag]['ci']['grid_ofli2'] = {int(T): ci(y, F[k, h, 3]) for h, T in enumerate(H) if T in (500, 1000)}
D['GAL'] = {}
for tag, fn in [('N1000', 'gal_1000.npz'), ('N500', 'gal_500.npz'), ('N2000', 'gal_2000.npz')]:
    d = np.load(fn); F, T, ref, H = d['F'], d['T'], d['ref'], d['H']; lab = np.where(ref < 1e-8, 1, np.where(ref > 1e-4, 0, -1)); k = lab >= 0; y = lab[k]
    sc = {'SALI': L(T[:, :, 0]), 'GALI3': L(T[:, :, 3]), 'OFLI': T[:, :, 1], 'MEGNO': T[:, :, 2], 'OFLI2': T[:, :, 4], 'bend2': T[:, :, 5],
          'grid_stretch': F[:, :, 0], 'grid_fold': F[:, :, 1], 'grid_ofli2': F[:, :, 2]}
    D['GAL'][tag] = {'n': int(k.sum()), 'chaotic_frac': float(y.mean()), 'excluded': int((lab < 0).sum()), 'h': [float(d['h1']), float(d['h2'])], 'T': H.tolist(),
                     'auc': {m: [float(auc(y, v[k, h])) for h in range(len(H))] for m, v in sc.items()},
                     'ci': {m: {int(t): ci(y, v[k, h]) for h, t in enumerate(H) if t in (100, 200)} for m, v in sc.items()} if tag == 'N1000' else {}}
json.dump(D, open('paper/numbers.json', 'w'), indent=1)
for m in ('SALI', 'OFLI2', 'bend2', 'grid_ofli2'):
    print('GAL', m, [round(x, 3) for x in D['GAL']['N1000']['auc'][m]], D['GAL']['N1000']['ci'][m])
print('HH ci grid_ofli2', D['HH_E0.125']['ci']['grid_ofli2'], D['HH_E0.1']['ci']['grid_ofli2'])
print('3dof ci', {e: D['3DOF'][e]['ci']['grid_ofli2'] for e in ('E0.06', 'E0.07', 'E0.09')})
