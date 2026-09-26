"""Fig. 2: material probability contours around the regular orbit R (top) and the chaotic orbit C (bottom).
Reads ../data/fig2_contours.npz (from compute_fig2_contours.py)."""
import os, numpy as np, matplotlib.pyplot as plt
from style import DATA, TEXTWIDTH, save

d = np.load(os.path.join(DATA, 'fig2_contours.npz')); snaps = d['snaps']
fig, axs = plt.subplots(2, len(snaps), figsize=(TEXTWIDTH, 3.5))
for row, (name, col, desc) in enumerate((('R', '#2a78d6', 'regular'), ('C', '#e34948', 'chaotic'))):
    for j, n in enumerate(snaps):
        ax = axs[row, j]; p = d[f'{name}_{n}_pts']; ratio, N, t, _ = d[f'{name}_{n}_meta']
        ax.plot(np.r_[p[:, 0], p[:1, 0]], np.r_[p[:, 1], p[:1, 1]], color=col, lw=0.8)
        ax.set_title(f'{name} ({desc}), $n={n}$, $t={t:.0f}$\n$L/L_h={ratio:.2f}$, $N={int(N)}$', fontsize=7)
        ax.ticklabel_format(style='sci', scilimits=(-2, 2), useMathText=True); ax.tick_params(labelsize=7)
        ax.xaxis.get_offset_text().set_fontsize(6); ax.yaxis.get_offset_text().set_fontsize(6)
        if n == 0:
            ax.set_aspect('equal')
        if row == 1:
            ax.set_xlabel(r'$\delta y$')
        if j == 0:
            ax.set_ylabel(r'$\delta p_y$')
fig.tight_layout(h_pad=0.8); save(fig, 'fig2_contours')
