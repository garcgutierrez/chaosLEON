"""Fig. 11: accuracy (AUROC on the 25 711 labelled nodes of Fig. 4) versus the wall time of the complete 600^2
Henon-Heiles map, one panel per machine of data/machines.json (all hardware threads) and per GPU of data/gpu/ (fp64),
with shared axes.  Each marker is one integration time T.
Reads the time_to_accuracy.json and speedup.json files of every machine and data/gpu/<host>/gpu.json."""
import numpy as np, matplotlib.pyplot as plt
from style import COLWIDTH, TEXTWIDTH, STYLE, save
from time_to_accuracy import MACHINES, GPUS, ACC, wall

KEYS = [('SALI', 'SALI', 'SALI map'), ('LEON', 'LEON', 'tangent LEON map'), ('B_h', 'B_h', None),
        ('OFLI2_h', 'OFLI2_h', None), ('LEON_h', 'LEON_h', None)]
PAN = MACHINES + GPUS
M = len(PAN)
if M <= 2:
    fig, AX = plt.subplots(M, 1, figsize=(COLWIDTH, 2.9 if M == 1 else 5.3), sharex=True, sharey=True, squeeze=False)
    AX = AX[:, 0]
else:
    fig, AX = plt.subplots(1, M, figsize=(TEXTWIDTH, 2.75), sharex=True, sharey=True, squeeze=False)
    AX = AX[0]
for j, m in enumerate(PAN):
    ax = AX[j]; p = m['pmax']
    for name, sty, lab in KEYS:
        T = np.array(ACC['T'][name]); a = np.array(ACC['auroc'][name]); k = T >= 20
        l, c, mk, ls = STYLE[sty]
        if name == 'LEON':                           # tangent LEON: darker shade, to separate it from LEON_h
            c, mk = '#9c1f1e', 'P'
        ax.plot(wall(name, T[k], p, m), a[k], color=c, marker=mk, ls=ls, ms=3.5, lw=1.1, label=lab or l)
        for Tl in (50, 200, 1000):                   # annotate a few integration times on SALI and LEON_h
            if (name == 'SALI' or (name == 'LEON_h' and Tl < 1000)) and Tl in T:
                i = int(np.nonzero(T == Tl)[0][0])
                ax.annotate(f'$T={Tl}$', (wall(name, T[i], p, m), a[i]), textcoords='offset points',
                            xytext=(4, -9) if name == 'SALI' else (-6, 5), fontsize=6.5, color='#52514e',
                            ha='left' if name == 'SALI' else 'right')
    ax.set_xscale('log'); ax.set_ylim(0.6, 1.0); ax.grid(alpha=0.25, lw=0.4)
    if M > 1:
        what = 'fp64' if p == 'gpu' else f'$p={p}$ threads'
        ax.set_title(f"({'abcdefgh'[j]}) {m['label']}, {what}", loc='left', fontsize=7.5, pad=3)
    if M > 2 or j == M - 1:
        ax.set_xlabel('wall time of the $600^2$ map (s)' if M > 1 else
                      f'wall time of the $600^2$ map, {p} threads (s)')
    if j == 0 or M <= 2:
        ax.set_ylabel('AUROC')
AX[-1 if M > 2 else 0].legend(loc='lower right', fontsize=6.5)
fig.tight_layout(); save(fig, 'fig11_time_to_accuracy')
