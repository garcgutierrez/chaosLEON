import sys, numpy as np; sys.path.insert(0, '..')
from higher_order_indicators import scores, auc
cmp = [('OFLI4','OFLI2'),('Lam4','Lam2'),('Lam3','OFLI2'),('OFLI4_h','OFLI2_h(3pt)'),('Lam4_h','B_h(3pt)'),('eta3_h','OFLI2_h(3pt)')]
for fn in ['ho_hh_0125.npz','ho_hh_0100.npz','ho_3dof_006.npz','ho_3dof_007.npz','ho_3dof_009.npz','ho_gal.npz']:
    d = np.load(fn); ref = d['ref']; lab = np.where(ref<1e-8,1,np.where(ref>1e-4,0,-1)); k = lab>=0; y = lab[k]; sc = scores(d); H = d['H']
    rng = np.random.default_rng(0); i1 = np.where(y==1)[0]; i0 = np.where(y==0)[0]; B = [np.r_[rng.choice(i1,len(i1)), rng.choice(i0,len(i0))] for _ in range(300)]
    print(fn)
    for a, b in cmp:
        row = []
        for h in range(len(H)):
            za, zb = sc[a][k,h], sc[b][k,h]; dd = auc(y,za)-auc(y,zb)
            bs = [auc(y[j],za[j])-auc(y[j],zb[j]) for j in B]; lo, hi = np.percentile(bs,[2.5,97.5])
            row.append(f'{dd:+.3f}{"*" if lo>0 else ("-" if hi<0 else " ")}')
        print(f'  {a:>8s}-{b:<13s}', ' '.join(row), '  T=', list(H))
