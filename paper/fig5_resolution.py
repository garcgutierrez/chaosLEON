"""Fig. 5: dependence of the grid-as-ensemble indicators on the grid spacing.
Colour shade = grid size; dotted = B_h, solid with stars = OFLI2_h; dashed blue = tangent OFLI2; black = SALI.
Reads ../data/numbers.json (keys 'HH_resolution', 'HH_E0.125', '3DOF', 'GAL')."""
import os, json, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from style import DATA, TEXTWIDTH, STYLE, save

D = json.load(open(os.path.join(DATA, 'numbers.json')))
SHADES = {250: '#f5b27e', 500: '#eb6834', 1000: '#b8421a', 2000: '#6e2408', 4000: '#3a1203'}
fig, axs = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.9), sharey=True)

def pair(ax, T, fold, ofli2, N):
    ax.plot(T, fold, ls=':', marker='o', ms=2.5, color=SHADES[N])
    if ofli2 is not None:
        ax.plot(T, ofli2, ls='-', marker='*', ms=4, color=SHADES[N])

def ref(ax, c):
    ax.plot(c['T'], c['auc']['OFLI2'], color=STYLE['OFLI2'][1], ls='--', lw=1.1)
    ax.plot(c['T'], c['auc']['SALI'], color='k', lw=1.1)

R = D['HH_resolution']
for N in ('250', '500', '1000', '2000'):
    pair(axs[0], R[N]['T'], R[N]['fold'], R[N].get('grid_ofli2'), int(N))
ref(axs[0], D['HH_E0.125']); axs[0].set_title('(a) Hénon–Heiles, $E=1/8$')
C3 = D['3DOF']
for key, N in (('E0.07_N250', 250), ('E0.07', 1000), ('E0.07_N4000', 4000)):
    c = C3[key]; pair(axs[1], c['T'], c['auc']['grid_fold'], c['auc'].get('grid_ofli2'), N)
ref(axs[1], C3['E0.07']); axs[1].set_xscale('log'); axs[1].set_title('(b) 3-dof, $E=0.07$')
G = D['GAL']
for key, N in (('N500', 500), ('N1000', 1000), ('N2000', 2000)):
    c = G[key]; pair(axs[2], c['T'], c['auc']['grid_fold'], c['auc']['grid_ofli2'], N)
ref(axs[2], G['N1000']); axs[2].set_xscale('log'); axs[2].set_title('(c) galactic potential')
for ax in axs:
    ax.set_xlabel('$T$'); ax.grid(alpha=0.25, lw=0.4)
axs[0].set_ylabel('AUROC'); axs[0].set_ylim(0.5, 1.0)
handles = ([Line2D([], [], color='#52514e', ls=':', marker='o', ms=3, label=r'$\mathcal{B}_h$'),
            Line2D([], [], color='#52514e', ls='-', marker='*', ms=4, label=r'OFLI2$_h$'),
            Line2D([], [], color=STYLE['OFLI2'][1], ls='--', label='OFLI2 (tangent)'),
            Line2D([], [], color='k', label='SALI')]
           + [Line2D([], [], color=SHADES[N], lw=3, label=f'${N}^2$ grid') for N in (250, 500, 1000, 2000, 4000)])
fig.legend(handles=handles, loc='lower center', ncol=9, fontsize=7, bbox_to_anchor=(0.5, -0.01), handlelength=1.8, columnspacing=1.0)
fig.tight_layout(rect=(0, 0.07, 1, 1)); save(fig, 'fig5_resolution')
