"""Fig. 1: (a) covariance aspect ratio and SALI along the orbits C and R; (b) normalized aspect ratio vs SALI.
Reads ../data/fig1_aspect_ratio.npz (from compute_fig1_aspect_ratio.py)."""
import os, numpy as np, matplotlib.pyplot as plt
from style import DATA, TEXTWIDTH, save

d = np.load(os.path.join(DATA, 'fig1_aspect_ratio.npz')); t = d['t']
AR, EXPL, HAT, SPAIR, SALI = 0, 1, 2, 3, 4
RED, BLUE, GREY = '#e34948', '#2a78d6', '#8a8984'

fig, (a, b) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.4))
C, R = d['C'], d['R']
expl = np.where(C[:, EXPL] > 0, C[:, EXPL], 1e-17)                 # a collapse to zero is drawn at the axis floor
a.semilogy(t, expl, color=RED, ls=':', lw=0.8, label='C: AR, explicit Lyapunov update')
a.semilogy(t, C[:, AR], color=RED, lw=1.1, label='C: AR, square-root propagation')
a.semilogy(t, C[:, SALI], color='k', ls='--', lw=1.0, label='C: SALI')
a.semilogy(t, R[:, AR], color=BLUE, lw=1.1, label='R: AR, square-root propagation')
a.semilogy(t, R[:, SALI], color=GREY, ls='--', lw=1.0, label='R: SALI')
a.set_ylim(1e-17, 3); a.set_xlim(0, t[-1]); a.set_xlabel(r'$t$'); a.set_ylabel('value'); a.set_title('(a)')
a.legend(loc='lower left', fontsize=7, frameon=True, facecolor='white', edgecolor='none', framealpha=0.9)

s = np.logspace(-16, np.log10(1.99), 400)
for name, col, mk in (('R', BLUE, 'o'), ('S', '#1baf7a', 's'), ('C', RED, '^')):
    o = d[name]; k = o[:, SPAIR] > 0
    b.loglog(o[k, SPAIR], o[k, HAT], mk, color=col, ms=2, mew=0, label=name)
b.loglog(s, s/np.sqrt(4 - s**2), 'k-', lw=0.9, label=r'$\mathrm{SALI}/\sqrt{4-\mathrm{SALI}^2}$')
b.set_xlabel('SALI'); b.set_ylabel(r'$\widehat{\mathrm{AR}}$'); b.set_title('(b)'); b.legend(loc='upper left', fontsize=7)
fig.tight_layout(); save(fig, 'fig1_aspect_ratio')
