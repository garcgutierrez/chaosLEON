"""Fig. 9: paired AUROC differences (95% bootstrap CI) for the key claims in the 19 evaluable cases.
Reads ../data/robustness.json (from ho_runs/robustness.py)."""
import os, json, numpy as np, matplotlib.pyplot as plt
from style import DATA, TEXTWIDTH, save

R = json.load(open(os.path.join(DATA, 'robustness.json'))); R.pop('ext/gal_rz_085.npz')   # one chaotic orbit
TSM = {'hh': (50, 100), 'qo': (50, 100), '3dof': (200, 500), 'gal': (50, 200)}
ORDER = {'hh': 0, 'qo': 1, '3dof': 2, 'gal': 3}
PANELS = [('Lam4-Lam2', 0, r'LEON$\,-\,\Lambda_2$ ($T_s$)'), ('Lam4_h-B_h(3pt)', 0, r'LEON$_h-\mathcal{B}_h$ ($T_s$)'),
          ('Lam4-OFLI2', 1, r'LEON$\,-\,$OFLI2 ($T_m$)'), ('OFLI4-OFLI2', 0, r'OFLI$_4-$OFLI2 ($T_s$)'),
          ('OFLI2_h(3pt)-OFLI2', 1, r'OFLI2$_h-$OFLI2 ($T_m$)')]
BLUE, ORANGE = '#2a78d6', '#eb6834'
files = sorted(R, key=lambda f: (ORDER[R[f]['system']], R[f]['plane'] not in ('x0', 'q0', 'rz'), R[f]['E']))
lab = [R[f]['label'].replace(r'\bm q', r'\mathbf{q}').replace(r'\bm p_{12}', r'\mathbf{p}_{12}')
       + (r'$^\dagger$' if R[f]['n']*R[f]['chaotic'] < 100 else '') for f in files]
ys = np.arange(len(files))[::-1]
fig, axs = plt.subplots(1, len(PANELS), figsize=(TEXTWIDTH, 3.4), sharey=True)
for ax, (c, which, title) in zip(axs, PANELS):
    col = ORANGE if '_h' in c else BLUE; ax.axvline(0, color='#8a8984', lw=0.6)
    for y, f in zip(ys, files):
        d, lo, hi = R[f]['claims'][f'{c}@{TSM[R[f]["system"]][which]}']; sig = lo > 0 or hi < 0
        ax.plot([lo, hi], [y, y], color=col, lw=1.1, solid_capstyle='round')
        ax.plot(d, y, 'o', ms=3.5, mfc=col if sig else 'white', mec=col, mew=1.0)
    for i in range(1, len(files)):
        if R[files[i]]['system'] != R[files[i-1]]['system']:
            ax.axhline((ys[i] + ys[i-1])/2, color='#d6d5d0', lw=0.5)
    ax.set_title(title, fontsize=7); ax.set_xlabel(r'$\Delta$AUROC'); ax.grid(axis='x', alpha=0.25, lw=0.4)
    ax.tick_params(axis='x', labelsize=7)
axs[0].set_yticks(ys); axs[0].set_yticklabels(lab, fontsize=7)
fig.tight_layout(w_pad=0.6); save(fig, 'fig9_robustness')
