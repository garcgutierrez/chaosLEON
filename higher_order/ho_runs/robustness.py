"""Robustness summary over all cases (original six + extension runs): AUROC tables and paired-bootstrap
differences for the paper's key claims, at a short and an intermediate horizon per system."""
import sys, json, glob, numpy as np; sys.path.insert(0, '..')
from scipy.stats import rankdata
from higher_order_indicators import scores

CASES = [  # file, system, plane, label
 ('ho_hh_0125.npz', 'hh', 'x0', r'HH, $x=0$, $E=1/8$'), ('ho_hh_0100.npz', 'hh', 'x0', r'HH, $x=0$, $E=1/10$'),
 ('ext/hh_x0_0083.npz', 'hh', 'x0', r'HH, $x=0$, $E=1/12$'), ('ext/hh_x0_0140.npz', 'hh', 'x0', r'HH, $x=0$, $E=0.14$'),
 ('ext/hh_y0_0125.npz', 'hh', 'y0', r'HH, $y=0$, $E=1/8$'),
 ('ext/qo_x0_075.npz', 'qo', 'x0', r'QO, $x=0$, $E=0.75$'), ('ext/qo_x0_100.npz', 'qo', 'x0', r'QO, $x=0$, $E=1$'),
 ('ext/qo_x0_200.npz', 'qo', 'x0', r'QO, $x=0$, $E=2$'),
 ('ho_3dof_006.npz', '3dof', 'q0', r'3-dof, $\bm q=0$, $E=0.06$'), ('ho_3dof_007.npz', '3dof', 'q0', r'3-dof, $\bm q=0$, $E=0.07$'),
 ('ext/3dof_q0_008.npz', '3dof', 'q0', r'3-dof, $\bm q=0$, $E=0.08$'), ('ho_3dof_009.npz', '3dof', 'q0', r'3-dof, $\bm q=0$, $E=0.09$'),
 ('ext/3dof_p0_007.npz', '3dof', 'p0', r'3-dof, $\bm p_{12}=0$, $E=0.07$'), ('ext/3dof_p0_009.npz', '3dof', 'p0', r'3-dof, $\bm p_{12}=0$, $E=0.09$'),
 ('ho_gal.npz', 'gal', 'rz', r'Gal., $(\rho,z)$, $E=-0.75$'), ('ext/gal_rz_080.npz', 'gal', 'rz', r'Gal., $(\rho,z)$, $E=-0.80$'),
 ('ext/gal_rz_085.npz', 'gal', 'rz', r'Gal., $(\rho,z)$, $E=-0.85$'), ('ext/gal_rz_077.npz', 'gal', 'rz', r'Gal., $(\rho,z)$, $E=-0.77$'), ('ext/gal_rz_0745.npz', 'gal', 'rz', r'Gal., $(\rho,z)$, $E=-0.745$'), ('ext/gal_rpr_075.npz', 'gal', 'rpr', r'Gal., $(\rho,p_\rho)$, $E=-0.75$'),
]
TSM = {'hh': (50, 100), 'qo': (50, 100), '3dof': (200, 500), 'gal': (50, 200)}
CLAIMS = [('OFLI2', 'OFLI'), ('OFLI2', 'SALI'), ('OFLI2_h(3pt)', 'OFLI2'), ('OFLI4', 'OFLI2'), ('Lam4', 'Lam2'),
          ('Lam3', 'OFLI2'), ('Lam4_h', 'B_h(3pt)'), ('OFLI4_h', 'OFLI2_h(3pt)'), ('eta3_h', 'OFLI2_h(3pt)'), ('Lam4', 'OFLI2'), ('Lam4_h', 'OFLI2_h(3pt)')]


def auc(y, z):
    r = rankdata(z); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1*(n1 + 1)/2)/(n1*n0)


def auc_boot(y, z, B):
    """AUROC on each bootstrap index row of B."""
    out = np.empty(len(B))
    for i, b in enumerate(B):
        out[i] = auc(y[b], z[b])
    return out


res = {}
for fn, sysn, plane, lab in CASES:
    try:
        d = np.load(fn)
    except FileNotFoundError:
        print('missing', fn); continue
    ref = d['ref']; H = list(d['H'])
    labs = np.where(ref < 1e-8, 1, np.where(ref > 1e-4, 0, -1))
    fin = np.isfinite(ref) & np.all(np.isfinite(d['G']), axis=(1, 2)) & np.all(np.isfinite(d['Tn']), axis=(1, 2, 3))
    k = (labs >= 0) & fin; y = labs[k]; sc = scores(d)
    tab = {nm: [float(auc(y, v[k, h])) for h in range(len(H))] for nm, v in sc.items()}
    rng = np.random.default_rng(0); i1 = np.where(y == 1)[0]; i0 = np.where(y == 0)[0]
    B = [np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))] for _ in range(500)]
    cl = {}
    for T in TSM[sysn]:
        h = H.index(T)
        for a, b in CLAIMS:
            za, zb = sc[a][k, h], sc[b][k, h]; dd = auc_boot(y, za, B) - auc_boot(y, zb, B)
            cl[f'{a}-{b}@{T}'] = [tab[a][h] - tab[b][h], *np.percentile(dd, [2.5, 97.5]).tolist()]
    res[fn] = dict(system=sysn, plane=plane, label=lab, E=float(d['E']), n=int(k.sum()), chaotic=float(y.mean()),
                   excluded_nonfinite=int((~fin).sum()), h=[float(d['h1']), float(d['h2'])], H=[int(t) for t in H], auc=tab, claims=cl)
    print(f'{fn:24s} n={k.sum():4d} chaotic={y.mean():.2f} nonfinite={(~fin).sum()}')
    for T in TSM[sysn]:
        h = H.index(T)
        print(f'   T={T:4d} SALI {tab["SALI"][h]:.3f} OFLI {tab["OFLI"][h]:.3f} OFLI2 {tab["OFLI2"][h]:.3f} B {tab["Lam2"][h]:.3f} '
              f'L3 {tab["Lam3"][h]:.3f} L4 {tab["Lam4"][h]:.3f} | OFLI2h {tab["OFLI2_h(3pt)"][h]:.3f} Bh {tab["B_h(3pt)"][h]:.3f} '
              f'L4h {tab["Lam4_h"][h]:.3f} eta3h {tab["eta3_h"][h]:.3f}')
        print('          ' + '  '.join(f'{c.split("@")[0]}:{v[0]:+.3f}{"*" if v[1] > 0 else ("-" if v[2] < 0 else " ")}'
                                       for c, v in cl.items() if c.endswith(f'@{T}')))
json.dump(res, open('robustness.json', 'w'), indent=1)
