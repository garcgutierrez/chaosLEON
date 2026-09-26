"""Fig. 8: AUROC gain of fourth- over second-order indicators versus integration time (six original cases).
Reads ../data/higher_order_results.json (from higher_order_indicators.py analyze)."""
import os, json, numpy as np, matplotlib.pyplot as plt
from style import DATA, TEXTWIDTH, save

R = json.load(open(os.path.join(DATA, 'higher_order_results.json')))
CASES = [('ho_hh_0125.npz', 'Hénon–Heiles, $E=1/8$'), ('ho_hh_0100.npz', 'Hénon–Heiles, $E=1/10$'),
         ('ho_gal.npz', 'galactic, $E=-0.75$'), ('ho_3dof_006.npz', '3-dof, $E=0.06$'),
         ('ho_3dof_007.npz', '3-dof, $E=0.07$'), ('ho_3dof_009.npz', '3-dof, $E=0.09$')]
BLUE, RED = '#2a78d6', '#e34948'
SERIES = [('OFLI4', 'OFLI2', BLUE, '-', 'o', r'OFLI$_4-$OFLI2 (tangent)'),
          ('OFLI4_h', 'OFLI2_h(3pt)', BLUE, '--', 's', r'OFLI$_{4,h}-$OFLI2$_h$ (grid)'),
          ('Lam4', 'Lam2', RED, '-', 'o', r'LEON$\,-\,\Lambda_2$ (tangent)'),
          ('Lam4_h', 'B_h(3pt)', RED, '--', 's', r'LEON$_h-\mathcal{B}_h$ (grid)')]
fig, axs = plt.subplots(2, 3, figsize=(TEXTWIDTH, 3.9), sharey=True)
for ax, (fn, title) in zip(axs.flat, CASES):
    r = R[fn]; H = np.array(r['H']); ax.axhline(0, color='#8a8984', lw=0.6)
    for a, b, c, ls, mk, lab in SERIES:
        ax.plot(H, np.array(r['auc'][a]) - np.array(r['auc'][b]), color=c, ls=ls, marker=mk, ms=3, lw=1.1, label=lab)
    ax.set_xscale('log'); ax.set_xticks(H); ax.set_xticklabels([str(t) for t in H], fontsize=7); ax.minorticks_off()
    ax.set_title(title); ax.grid(alpha=0.25, lw=0.4)
for ax in axs[1]:
    ax.set_xlabel('integration time $T$')
for ax in axs[:, 0]:
    ax.set_ylabel(r'$\Delta$AUROC (order 4 $-$ order 2)')
axs[0, 0].legend(loc='upper right', fontsize=7)
fig.tight_layout(); save(fig, 'fig8_order_gain')
