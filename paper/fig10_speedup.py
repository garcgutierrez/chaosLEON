"""Fig. 10: strong-scaling study (Henon-Heiles, E = 1/8, 600^2 grid, T = 200) on every machine of data/machines.json.
Top row: wall time; bottom row: speed-up t_1/t_p, versus the number of threads p; one column per machine.
Reads the speedup.json files listed in ../data/machines.json (written by compute_speedup.py / run_benchmarks.py)."""
import numpy as np, matplotlib.pyplot as plt
from style import COLWIDTH, TEXTWIDTH, STYLE, save
from time_to_accuracy import MACHINES

SER = [('sali', 'SALI map', STYLE['SALI'][1], 'o', '-'),
       ('leon', 'tangent LEON map', STYLE['LEON'][1], 'P', '--'),
       ('grid', r'grid maps ($\mathcal{S}_h$, $\mathcal{B}_h$, OFLI2$_h$, LEON$_h$)', STYLE['OFLI2_h'][1], '*', '-'),
       ('grid_adv', 'grid: advance phase only', STYLE['S_h'][1], '.', ':')]
M = len(MACHINES)
fig, AX = plt.subplots(2, M, figsize=(COLWIDTH if M == 1 else TEXTWIDTH, 5.0 if M == 1 else 4.4),
                       sharex='col', squeeze=False)
for j in range(1, M):
    AX[0, j].sharey(AX[0, 0])
smax = max(m['pmax'] for m in MACHINES)
for j, m in enumerate(MACHINES):
    ax1, ax2 = AX[0, j], AX[1, j]; P = np.array(m['P']); W = m['SP']; nc = m['n_cores']
    for key, lab, c, mk, ls in SER:
        t = np.array([W[key][str(p)] for p in P])
        ax1.plot(P, t, color=c, marker=mk, ls=ls, label=lab, ms=4.5 if mk != '.' else 7, lw=1.1)
        ax2.plot(P, t[0]/t, color=c, marker=mk, ls=ls, ms=4.5 if mk != '.' else 7, lw=1.1)
    for ax in (ax1, ax2):
        if m['pmax'] > nc:
            ax.axvspan(nc, P[-1] + 0.5, color='#8a8984', alpha=0.10, lw=0)
        ax.grid(alpha=0.25, lw=0.4)
    ax2.plot([1, P[-1]], [1, P[-1]], color='#8a8984', lw=0.8, ls='-')
    ax2.set_ylim(0, P[-1] + 0.5)
    if m['pmax'] > nc:
        ht = '(2 per core)' if m['pmax'] == 2*nc else m.get('smt_note', '(SMT)')
        ax2.text(nc + 0.02*P[-1], 0.065*P[-1], 'hyper-threads\n' + ht, color='#52514e', fontsize=6.5, ha='left', va='bottom')
    ax2.set_xlabel('number of threads $p$')
    ticks = P if len(P) <= 10 else P[::2] if P[-1] in P[::2] else list(P[::2]) + [P[-1]]
    ax2.set_xticks(ticks); ax2.set_xlim(0.6, P[-1] + 0.4)
    if M > 1:
        cn = f" ({m['core_note']})" if m.get('core_note') else ''
        ax1.set_title(f"{m['label']}: {nc} cores{cn}, {m['n_logical']} threads", fontsize=8)
    ax1.text(-0.2 if j == 0 else -0.08, 1.0, '(%s)' % 'abcdefgh'[j], transform=ax1.transAxes, fontsize=8)
    ax2.text(-0.2 if j == 0 else -0.08, 1.0, '(%s)' % 'abcdefgh'[M + j], transform=ax2.transAxes, fontsize=8)
AX[0, 0].set_yscale('log'); AX[0, 0].set_ylabel('wall time (s)'); AX[1, 0].set_ylabel(r'speed-up $t_1/t_p$')
AX[0, -1].legend(loc='upper right', fontsize=6.5, handlelength=2.6)
fig.tight_layout()
for j, m in enumerate(MACHINES):                  # label of the ideal line, rotated to its on-screen slope
    ax2 = AX[1, j]; x = 0.78*m['pmax']; a = ax2.transData.transform([(1, 1), (m['pmax'], m['pmax'])])
    ang = np.degrees(np.arctan2(a[1, 1] - a[0, 1], a[1, 0] - a[0, 0]))
    ax2.text(x, x*1.03, r'ideal, $t_1/t_p=p$', color='#52514e', fontsize=7, ha='right', va='bottom', rotation=ang,
             rotation_mode='anchor')
save(fig, 'fig10_speedup')
