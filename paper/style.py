"""Shared plotting style for the paper figures (import from every fig*.py script).

Edit COLORS / LABELS / MARKERS here to restyle all figures at once.
Figures are written to ../figs/<name>.pdf (and a .png preview next to it).
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
FIGS = os.path.join(HERE, '..', 'figs')

TEXTWIDTH = 7.0      # in, two-column REVTeX page width
COLWIDTH = 3.4       # in, one column

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.2, 'lines.markersize': 3.5,
    'axes.edgecolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

# indicator key -> (label, color, marker, linestyle)
# first-order: greys/violet; second-order tangent: blues/aqua; grid: warm colours; LEON: red.
STYLE = {
    'SALI':    (r'SALI', '#0b0b0b', 'o', '-'),
    'GALI3':   (r'GALI$_3$', '#8a8984', 's', '--'),
    'MEGNO':   (r'$\langle Y\rangle$ (MEGNO)', '#b5b4ae', 'v', ':'),
    'OFLI':    (r'OFLI', '#4a3aa7', '^', '-'),
    'OFLI2':   (r'OFLI2', '#2a78d6', 'D', '--'),
    'B':       (r'$\mathcal{B}$ (tangent)', '#1baf7a', 'x', '-.'),
    'LEON':    (r'LEON', '#e34948', 'P', '-'),
    'S_h':     (r'$\mathcal{S}_h$ (grid)', '#eda100', '.', ':'),
    'B_h':     (r'$\mathcal{B}_h$ (grid)', '#eb6834', 'x', '-.'),
    'OFLI2_h': (r'OFLI2$_h$ (grid)', '#8f3fb5', '*', '-'),
    'LEON_h':  (r'LEON$_h$ (grid)', '#e34948', 'P', '--'),
    'UFR':     (r'UFR contour, $\sigma=10^{-3}$', '#e87ba4', 'P', 'none'),
}


def line(ax, x, y, key, **kw):
    lab, c, m, ls = STYLE[key]
    opts = dict(color=c, marker=m, ls=ls, label=lab, ms=3.5, lw=1.1)
    opts.update(kw)
    return ax.plot(x, y, **opts)


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, name + '.pdf'))
    fig.savefig(os.path.join(FIGS, name + '.png'), dpi=200)
    print('wrote', os.path.normpath(os.path.join(FIGS, name + '.pdf')))
