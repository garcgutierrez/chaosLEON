"""Fig. 6: AUROC versus integration time for the 3-dof Hamiltonian at E = 0.06, 0.07, 0.09.
Reads ../data/numbers.json (key '3DOF')."""
import os, json, matplotlib.pyplot as plt
from style import DATA, TEXTWIDTH, line, save

D = json.load(open(os.path.join(DATA, 'numbers.json')))['3DOF']
KEYS = [('SALI', 'SALI'), ('GALI3', 'GALI3'), ('OFLI', 'OFLI'), ('MEGNO', 'MEGNO'), ('OFLI2', 'OFLI2'),
        ('bend2', 'B'), ('grid_stretch', 'S_h'), ('grid_fold', 'B_h'), ('grid_ofli2', 'OFLI2_h')]
fig, axs = plt.subplots(1, 3, figsize=(TEXTWIDTH, 3.0), sharey=True)
for ax, (case, lab) in zip(axs, (('E0.06', '(a) $E=0.06$'), ('E0.07', '(b) $E=0.07$'), ('E0.09', '(c) $E=0.09$'))):
    c = D[case]
    for k, sty in KEYS:
        line(ax, c['T'], c['auc'][k], sty)
    ax.set_xscale('log'); ax.set_title(f'{lab} ({100*c["chaotic_frac"]:.0f}% chaotic)')
    ax.set_xlabel('integration time $T$'); ax.set_ylim(0.4, 1.0); ax.grid(alpha=0.25, lw=0.4)
axs[0].set_ylabel('AUROC'); axs[2].legend(loc='lower right', fontsize=7)
fig.tight_layout(); save(fig, 'fig6_3dof_auroc')
