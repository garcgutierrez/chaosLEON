"""Fig. 4: complete Henon-Heiles chaos maps at T = 200 (600^2 grid): SALI, reference SALI(1e4), B_h, OFLI2_h, LEON_h.
Reads ../data/fig4_maps.npz (from compute_fig4_maps.py)."""
import os, numpy as np, matplotlib.pyplot as plt
from style import DATA, TEXTWIDTH, save

d = np.load(os.path.join(DATA, 'fig4_maps.npz')); y, py = d['y'], d['py']
ext = [y[0], y[-1], py[0], py[-1]]; ts, tg = float(d['t_sali']), float(d['t_grid'])
fig, axs = plt.subplots(1, 5, figsize=(TEXTWIDTH, 1.85), sharey=True, layout='constrained')

def show(ax, m, title, cmap, vmin, vmax, clab):
    im = ax.imshow(m.T, origin='lower', extent=ext, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', interpolation='nearest')
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.01); cb.ax.tick_params(labelsize=6.5)
    ax.set_title(title, fontsize=6.5); ax.set_xlabel('$y$', fontsize=7); ax.tick_params(labelsize=6.5)

show(axs[0], np.log10(np.maximum(d['SALI'], 1e-16)), r'(a) $\log_{10}$SALI, $T=200$', 'viridis_r', -16, 0, r'$\log_{10}$SALI')
ij = d['ref_ij']; ref = np.log10(np.maximum(d['ref'], 1e-16))
axs[1].scatter(y[ij[:, 0]], py[ij[:, 1]], c=ref, s=0.15, cmap='viridis_r', vmin=-16, vmax=0, marker='s', lw=0)
axs[1].set_xlim(ext[:2]); axs[1].set_ylim(ext[2:]); axs[1].set_title(r'(b) reference, $t=10^4$', fontsize=6.5)
axs[1].set_xlabel('$y$', fontsize=7); axs[1].tick_params(labelsize=6.5)
sm = plt.cm.ScalarMappable(cmap='viridis_r', norm=plt.Normalize(-16, 0)); cb = fig.colorbar(sm, ax=axs[1], fraction=0.06, pad=0.01)
cb.ax.tick_params(labelsize=6.5)
for ax, key, lab, q in ((axs[2], 'B_h', r'(c) $\mathcal{B}_h$', (2, 98)), (axs[3], 'OFLI2_h', r'(d) OFLI2$_h$', (2, 98)),
                        (axs[4], 'LEON_h', r'(e) LEON$_h$', (2, 98))):
    m = d[key]; lo, hi = np.nanpercentile(m, q)
    show(ax, m, f'{lab}, $T=200$', 'magma', lo, hi, lab.split(' ', 1)[1])
axs[0].set_ylabel('$p_y$', fontsize=7)
save(fig, 'fig4_maps')   # wall times (t_sali, t_grid) are quoted in the caption
