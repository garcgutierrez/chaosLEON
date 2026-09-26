"""Fig. 7: AUROC versus integration time for the galactic potential (1000^2 grid).
Reads ../data/numbers.json (key 'GAL' -> 'N1000')."""
import os, json, matplotlib.pyplot as plt
from style import DATA, COLWIDTH, line, save

c = json.load(open(os.path.join(DATA, 'numbers.json')))['GAL']['N1000']
KEYS = [('SALI', 'SALI'), ('GALI3', 'GALI3'), ('OFLI', 'OFLI'), ('MEGNO', 'MEGNO'), ('OFLI2', 'OFLI2'),
        ('bend2', 'B'), ('grid_stretch', 'S_h'), ('grid_fold', 'B_h'), ('grid_ofli2', 'OFLI2_h')]
fig, ax = plt.subplots(figsize=(COLWIDTH, 3.2))
for k, sty in KEYS:
    line(ax, c['T'], c['auc'][k], sty)
ax.set_xscale('log'); ax.set_xlabel('integration time $T$'); ax.set_ylabel('AUROC'); ax.set_ylim(0.55, 1.0)
ax.grid(alpha=0.25, lw=0.4); ax.legend(loc='lower right', fontsize=7)
fig.tight_layout(); save(fig, 'fig7_galactic')
