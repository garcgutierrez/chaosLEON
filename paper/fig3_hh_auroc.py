"""Fig. 3: AUROC versus integration time for the Henon-Heiles system at E = 1/8 and E = 1/10.
Reads ../data/numbers.json (keys 'HH_E0.125', 'HH_E0.1', 'UFR')."""
import os, json, matplotlib.pyplot as plt
from style import DATA, TEXTWIDTH, line, save

D = json.load(open(os.path.join(DATA, 'numbers.json')))
KEYS = [('SALI', 'SALI'), ('GALI3', 'GALI3'), ('OFLI', 'OFLI'), ('MEGNO', 'MEGNO'), ('OFLI2', 'OFLI2'),
        ('bend2', 'B'), ('grid_stretch', 'S_h'), ('grid_fold', 'B_h'), ('grid_ofli2', 'OFLI2_h')]
fig, axs = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.7), sharey=True)
for ax, (case, title) in zip(axs, (('HH_E0.125', '(a) $E=1/8$'), ('HH_E0.1', '(b) $E=1/10$'))):
    c = D[case]
    for k, sty in KEYS:
        line(ax, c['T'], c['auc'][k], sty)
    if case == 'HH_E0.125':
        u = D['UFR']['1e-3']; line(ax, u['T'], u['UFR'], 'UFR', ms=5)
    ax.set_title(f'{title} ($n={c["n"]}$, {100*c["chaotic_frac"]:.0f}% chaotic)')
    ax.set_xlabel('integration time $T$'); ax.set_ylim(0.45, 1.0); ax.grid(alpha=0.25, lw=0.4)
axs[0].set_ylabel('AUROC'); axs[0].legend(loc='lower right', ncol=1, fontsize=7)
fig.tight_layout(); save(fig, 'fig3_hh_auroc')
