"""LaTeX rows of Table V (higher order, six original cases), Table VI (robustness, additional cases) and Table VII
(wall times, one block per machine), and the numbers quoted in Sec. VI G.
Reads ../data/higher_order_results.json, ../data/robustness.json and the timing files of ../data/machines.json;
writes ../tables/tab_higher_rows.tex, tab_robust_rows.tex, tab_time_rows.tex and parallel_macros.tex (numbers of Sec. VI G),
which main.tex includes with \\input.  Column order must match the table headers."""
import os, json, math
from style import DATA
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tables'); os.makedirs(OUT, exist_ok=True)
fm = lambda x, b: (r'\textbf{%.3f}' % x) if abs(x - b) < 5e-4 else '%.3f' % x

# ---- Table V: SALI | OFLI2 OFLI4 B L3 LEON eta2 | OFLI2_h OFLI4_h B_h LEON_h eta3_h
R = json.load(open(os.path.join(DATA, 'higher_order_results.json')))
cases = [('ho_hh_0125.npz', r'HH, $E=1/8$', [50, 100, 200]), ('ho_hh_0100.npz', r'HH, $E=1/10$', [50, 100, 200]),
         ('ho_3dof_006.npz', r'3-dof, $E=0.06$', [200, 500, 1000]), ('ho_3dof_007.npz', r'3-dof, $E=0.07$', [100, 200, 500]),
         ('ho_3dof_009.npz', r'3-dof, $E=0.09$', [100, 200, 500]), ('ho_gal.npz', r'Galactic', [50, 100, 200, 500])]
tan = ['SALI', 'OFLI2', 'OFLI4', 'Lam2', 'Lam3', 'Lam4', 'eta2']; grid = ['OFLI2_h(3pt)', 'OFLI4_h', 'B_h(3pt)', 'Lam4_h', 'eta3_h']
rows = []
for fn, name, Ts in cases:
    r = R[fn]; H = r['H']
    for i, T in enumerate(Ts):
        h = H.index(T); vt = [r['auc'][c][h] for c in tan]; vg = [r['auc'][c][h] for c in grid]
        bt, bg = max(vt[1:]), max(vg)
        rows.append(f'{name if i == 0 else ""} & {T} & ' + ' & '.join(['%.3f' % vt[0]] + [fm(x, bt) for x in vt[1:]] + [fm(x, bg) for x in vg]) + r'\\')
    if fn != cases[-1][0]:
        rows.append(r'\hline')
open(os.path.join(OUT, 'tab_higher_rows.tex'), 'w').write('\n'.join(rows) + '\n')

# ---- Table VI: n_ch T SALI | OFLI OFLI2 B L3 LEON | OFLI2_h B_h LEON_h eta3_h
Q = json.load(open(os.path.join(DATA, 'robustness.json')))
TSM = {'hh': (50, 100), 'qo': (50, 100), '3dof': (200, 500), 'gal': (50, 200)}; ORDER = {'hh': 0, 'qo': 1, '3dof': 2, 'gal': 3}
new = sorted([f for f in Q if f.startswith('ext/') and f != 'ext/gal_rz_085.npz'],
             key=lambda f: (ORDER[Q[f]['system']], Q[f]['plane'] not in ('x0', 'q0', 'rz'), Q[f]['E']))
tcols = ['SALI', 'OFLI', 'OFLI2', 'Lam2', 'Lam3', 'Lam4']; gcols = ['OFLI2_h(3pt)', 'B_h(3pt)', 'Lam4_h', 'eta3_h']
rows = []; prev = None
for f in new:
    v = Q[f]; nch = round(v['n']*v['chaotic'])
    if prev is not None and v['system'] != prev:
        rows.append(r'\hline')
    prev = v['system']
    for i, T in enumerate(TSM[v['system']]):
        h = v['H'].index(T); t = [v['auc'][c][h] for c in tcols]; g = [v['auc'][c][h] for c in gcols]
        head = (v['label'] + (r'$^\dagger$' if nch < 100 else '') + f' & {nch}') if i == 0 else ' & '
        rows.append(f'{head} & {T} & ' + ' & '.join(['%.3f' % t[0]] + [fm(x, max(t[1:])) for x in t[1:]] + [fm(x, max(g)) for x in g]) + r'\\')
open(os.path.join(OUT, 'tab_robust_rows.tex'), 'w').write('\n'.join(rows) + '\n')
print('wrote tables/tab_higher_rows.tex, tables/tab_robust_rows.tex')

# ---- Table VII: wall time of the complete 600^2 Henon-Heiles maps at fixed T and at equal accuracy, one block per
# machine of data/machines.json; and tables/parallel_macros.tex with the numbers quoted in Sec. VI G (\pv{A}{key}).
from time_to_accuracy import MACHINES, GPUS, ACC, wall, t_star, speedup, auroc_maxdiff
import numpy as np
LAB = [('SALI', 'SALI'), ('LEON', r'$\LEON$ (tangent)'), ('S_h', r'$\mathcal S_h$ (grid)'), ('B_h', r'$\mathcal B_h$ (grid)'),
       ('OFLI2_h', r'$\mathrm{OFLI2}_h$ (grid)'), ('LEON_h', r'$\LEON_h$ (grid)')]
TARGETS = [0.90, 0.95, 0.97]
wt = lambda t: '%.0f' % t if t >= 10 else '%.1f' % t
rows = []
for j, m in enumerate(MACHINES):
    if len(MACHINES) > 1:
        if j:
            rows.append(r'\hline')
        rows.append(r'\multicolumn{13}{l}{\emph{(%s) %s: %d cores, $p=%d$ threads}}\\' % ('abcdefgh'[j], m['label'], m['n_cores'], m['pmax']))
    for name, lab in LAB:
        a200 = ACC['auroc'][name][ACC['T'][name].index(200.0)]
        cells = [lab, '%.3f' % a200, wt(wall(name, 200.0, 1, m)), wt(wall(name, 200.0, 'max', m))]
        for A in TARGETS:
            Ts = t_star(name, A)
            cells += ['---']*3 if Ts is None else ['%.0f' % Ts, wt(wall(name, Ts, 1, m)), wt(wall(name, Ts, 'max', m))]
        rows.append(' & '.join(cells) + r'\\')
open(os.path.join(OUT, 'tab_time_rows.tex'), 'w').write('\n'.join(rows) + '\n')
print('wrote tables/tab_time_rows.tex')

f1 = lambda x: '%.1f' % x
f2 = lambda x: '%.2f' % x
pc = lambda x: '%.0f' % (100*x)


def osname(s):
    if s.startswith('Windows'):                   # Python < 3.12 reports Windows 11 as 'Windows-10-10.0.<build>'
        w = s.split('-'); ver = w[1] if len(w) > 1 else ''
        try:
            if ver == '10' and int(w[2].split('.')[2]) >= 22000:
                ver = '11'
        except (IndexError, ValueError):
            pass
        return 'Windows~' + ver
    if s.startswith('macOS') or s.startswith('Darwin'):
        return 'macOS'
    return 'Linux' if s.startswith('Linux') else s


def pv(m, A=None):
    """Numbers of Sec. VI G for machine m (A: reference machine, for the cross-machine ratios)."""
    SP = m['SP']; nc = m['n_cores']; pM = m['pmax']
    pC = max(p for p in m['P'] if p <= nc)                      # one thread per physical core
    t = lambda k, p: SP[k][str(p)]
    d = dict(name=m['label'], cores=str(nc), threads=str(pM), pC=str(pC),
             clock=('%.1f' % m['clock_ghz']) if m.get('clock_ghz') else '?', numba='.'.join(m.get('numba', '?').split('.')[:2]),
             layer=m['layer'].upper() if m['layer'] in ('tbb', 'omp') else m['layer'], osname=osname(m['os']))
    for k, s in (('sali', 'sali'), ('leon', 'leon'), ('grid', 'grid'), ('grid_adv', 'adv')):
        d['t1' + s] = wt(t(k, 1)); d['tM' + s] = wt(t(k, pM)); d['tC' + s] = wt(t(k, pC))
        d['sC' + s] = f1(speedup(m, k, pC)); d['sM' + s] = f1(speedup(m, k, pM))
    worst = min(speedup(m, k, pC) for k in ('sali', 'leon', 'grid'))
    d['effC'] = f2(worst/pC); d['serialC'] = f1(100*max(pC/worst - 1, 0)/(pC - 1)) if pC > 1 else '0'
    for p, s in ((1, '1'), (pC, 'C'), (pM, 'M')):
        r = t('grid', p) - t('grid_adv', p); d['read' + s] = wt(r); d['share' + s] = pc(r/t('grid', p))
    g = [t(k, pC)/t(k, pM) - 1 for k in ('sali', 'leon')]
    d['htlo'], d['hthi'] = pc(min(g)), pc(max(g)); d['htgrid'] = pc(t('grid', pC)/t('grid', pM) - 1)
    d['sCrows'] = f1(t('grid_rows', 1)/t('grid_rows', pC)) if str(pC) in SP.get('grid_rows', {}) else '?'
    d['q1'] = f2(t('grid', 1)/t('sali', 1)); d['qM'] = f2(t('grid', pM)/t('sali', pM))
    rep = [abs(m['repeat'][p][k]/t(k, int(p)) - 1) for p in m['repeat'] for k in m['repeat'][p]]
    d['rep'] = pc(max(rep)) if rep else '?'
    # equal accuracy: LEON_h at A = 0.97 and 0.95, tangent LEON at 0.97, accuracy limits
    for A_, s in ((0.97, '97'), (0.95, '95')):
        Ts, Tl = t_star('SALI', A_), t_star('LEON_h', A_)
        d['t1sali' + s] = wt(wall('SALI', Ts, 1, m)); d['tMsali' + s] = wt(wall('SALI', Ts, pM, m))
        d['t1leonh' + s] = wt(wall('LEON_h', Tl, 1, m)); d['tMleonh' + s] = wt(wall('LEON_h', Tl, pM, m))
        d['g1' + s] = f1(wall('SALI', Ts, 1, m)/wall('LEON_h', Tl, 1, m))
        d['gM' + s] = f1(wall('SALI', Ts, pM, m)/wall('LEON_h', Tl, pM, m))
        if s == '97':
            d['fT'] = f1(Ts/Tl)
            d['fc'] = f1((wall('SALI', Ts, 1, m)/Ts)/(wall('LEON_h', Tl, 1, m)/Tl))
            d['fS'] = f1((wall('LEON_h', Tl, 1, m)/wall('LEON_h', Tl, pM, m))/(wall('SALI', Ts, 1, m)/wall('SALI', Ts, pM, m)))
            d['comb'] = '%.0f' % (wall('SALI', Ts, 1, m)/wall('LEON_h', Tl, pM, m))
            To = t_star('OFLI2_h', A_)
            d['g1ofli97'] = f1(wall('SALI', Ts, 1, m)/wall('OFLI2_h', To, 1, m)); d['gMofli97'] = f1(wall('SALI', Ts, pM, m)/wall('OFLI2_h', To, pM, m))
            Tt = t_star('LEON', A_)
            d['g1leon'] = f1(wall('SALI', Ts, 1, m)/wall('LEON', Tt, 1, m)); d['gMleon'] = f1(wall('SALI', Ts, pM, m)/wall('LEON', Tt, pM, m))
    d['tMleonhmax'] = wt(wall('LEON_h', t_star('LEON_h', 0.980), pM, m))
    d['tMleon98'] = wt(wall('LEON', t_star('LEON', 0.98), pM, m)); d['tMsali98'] = wt(wall('SALI', t_star('SALI', 0.98), pM, m))
    if m.get('RS'):                                             # one grid indicator at a time (compute_readout_split.py)
        R = m['RS']
        for p, sfx in ((1, '1'), (pM, 'M')):
            adv = R['grid_adv'][str(p)]
            d['rs' + sfx + 'adv'] = wt(adv); d['rs' + sfx + 'all'] = wt(R['grid_all'][str(p)])
            d['rs' + sfx + 'ovall'] = pc(R['grid_all'][str(p)]/adv - 1)
            ov = {k: R[k][str(p)]/adv - 1 for k in ('S_h', 'B_h', 'OFLI2_h', 'LEON_h')}
            for k, v in ov.items():
                d['rs' + sfx + 'ov' + k.replace('_h', '').replace('2', 'two')] = pc(v)
            d['rs' + sfx + 'ovlo'] = pc(min(ov.values())); d['rs' + sfx + 'ovhi'] = pc(max(ov.values()))
    if A is not None:                                           # speed relative to the reference machine
        for k, s in (('sali', 'sali'), ('leon', 'leon'), ('grid', 'grid')):
            d['x1' + s] = f1(A['SP'][k]['1']/t(k, 1)); d['xM' + s] = f1(A['SP'][k][str(A['pmax'])]/t(k, pM))
        d['xequal'] = f1(wall('LEON_h', t_star('LEON_h', 0.97), 'max', A)/wall('LEON_h', t_star('LEON_h', 0.97), pM, m))
        d['accdiff'] = '%.0e' % auroc_maxdiff(m) if auroc_maxdiff(m) > 0 else '0'
    return d


ft = lambda t: '%.0f' % t if t >= 10 else '%.1f' % t if t >= 1 else '%.2f' % t


def gpu_macros(g, B, A):
    r64, r32 = g['raw']['fp64']['600'], g['raw']['fp32']['600']; W, V = r64['times'], r32['times']
    d = dict(name=g['label'], cc=g['gpu']['compute_capability'], sm=str(g['gpu']['multiprocessors']),
             mem='%.0f' % g['gpu']['memory_GB'])
    for k in ('sali', 'leon', 'grid', 'grid_adv'):
        s = k.replace('grid_adv', 'adv')
        d['t64' + s] = ft(W[k]); d['t32' + s] = ft(V[k])
        if k != 'grid_adv':
            d['rB' + s] = f1(B['SP'][k][str(B['pmax'])]/W[k]); d['rA' + s] = f1(A['SP'][k][str(A['pmax'])]/W[k])
    rr = [W[k]/V[k] for k in ('sali', 'leon', 'grid')]
    d['r32lo'], d['r32hi'] = '%.0f' % min(rr), '%.0f' % max(rr)
    d['share64'] = '%.0f' % (100*(W['grid'] - W['grid_adv'])/W['grid'])
    one = [W['grid_' + k]/W['grid_adv'] - 1 for k in ('S_h', 'B_h', 'OFLI2_h', 'LEON_h')]
    d['onelo'], d['onehi'] = '%.1f' % (100*min(one)), '%.1f' % (100*max(one))
    if '1200' in g['raw']['fp64']:
        W2 = g['raw']['fp64']['1200']['times']; d['n1200nodes'] = '%d' % g['raw']['fp64']['1200']['n_nodes']
        d['x1200grid'] = f2(W2['grid']/W['grid']); d['x1200sali'] = f2(W2['sali']/W['sali'])
        d['t1200grid64'] = ft(W2['grid']); d['t1200grid32'] = ft(g['raw']['fp32']['1200']['times']['grid'])
    c = r64['check']; e = max(c[k + '_maxdiff'] for k in ('S_h', 'B_h', 'OFLI2_h', 'LEON_h'))
    d['maxdiff64'] = r'%d\times10^{%d}' % (round(e/10**math.floor(math.log10(e))), math.floor(math.log10(e)))
    c32 = r32['check']; d['maxdiff32'] = '%.1f' % max(c32[k + '_maxdiff'] for k in ('S_h', 'B_h', 'OFLI2_h', 'LEON_h'))
    au = [abs(r32['auroc'][k] - r64['auroc'][k]) for k in ('SALI', 'LEON', 'S_h', 'B_h', 'OFLI2_h', 'LEON_h')]
    d['aucdiff32'] = '%.1f' % (1e4*max(au))                    # in units of 1e-4
    Ts, Tl, Tt, To = t_star('SALI', 0.97), t_star('LEON_h', 0.97), t_star('LEON', 0.97), t_star('OFLI2_h', 0.97)
    d['tsali97'] = ft(wall('SALI', Ts, 'gpu', g)); d['tleonh97'] = ft(wall('LEON_h', Tl, 'gpu', g))
    d['tleon97'] = ft(wall('LEON', Tt, 'gpu', g)); d['tofli97'] = ft(wall('OFLI2_h', To, 'gpu', g))
    d['g97'] = f1(wall('SALI', Ts, 'gpu', g)/wall('LEON_h', Tl, 'gpu', g))
    d['gofli97'] = f1(wall('SALI', Ts, 'gpu', g)/wall('OFLI2_h', To, 'gpu', g))
    d['gleon97'] = f1(wall('SALI', Ts, 'gpu', g)/wall('LEON', Tt, 'gpu', g))
    d['combA'] = '%.0f' % (wall('SALI', Ts, 1, A)/wall('LEON_h', Tl, 'gpu', g))
    d['combB'] = '%.0f' % (wall('SALI', Ts, 1, B)/wall('LEON_h', Tl, 'gpu', g))
    d['xB97'] = f1(wall('LEON_h', Tl, 'max', B)/wall('LEON_h', Tl, 'gpu', g))
    d['xBsali97'] = f1(wall('SALI', Ts, 'max', B)/wall('SALI', Ts, 'gpu', g))
    d['t32leonh200'] = ft(V['grid'])
    return d


if GPUS:                                                        # Table VIII: GPU
    g, B, A = GPUS[0], MACHINES[-1], MACHINES[0]; r64, r32 = g['raw']['fp64']['600'], g['raw']['fp32']['600']
    rows = []
    for name, lab in LAB:
        k = {'SALI': 'sali', 'LEON': 'leon'}.get(name, 'grid'); Ts = t_star(name, 0.97)
        cells = [lab, '%.3f' % r64['auroc'][name], '%.3f' % r32['auroc'][name], ft(r64['times'][k]), ft(r32['times'][k]),
                 ft(A['SP'][k][str(A['pmax'])]), ft(B['SP'][k][str(B['pmax'])])]
        cells += ['---']*3 if Ts is None else ['%.0f' % Ts, ft(wall(name, Ts, 'gpu', g)), ft(wall(name, Ts, 'max', B))]
        rows.append(' & '.join(cells) + r'\\')
    k = 'grid_adv'
    rows.append(' & '.join(['grid, advance only', '---', '---', ft(r64['times'][k]), ft(r32['times'][k]),
                            ft(A['SP'][k][str(A['pmax'])]), ft(B['SP'][k][str(B['pmax'])]), '---', '---', '---']) + r'\\')
    open(os.path.join(OUT, 'tab_gpu_rows.tex'), 'w').write('\n'.join(rows) + '\n')
    print('wrote tables/tab_gpu_rows.tex')

L = [r'% generated by scripts/make_tables.py from data/machines.json: numbers of Sec. VI G, used as \pv{machine}{key}',
     r'\makeatletter', r'\providecommand{\pv}[2]{\@ifundefined{pv@#1@#2}{\textbf{??}}{\csname pv@#1@#2\endcsname}}']
ref = pv(MACHINES[0]); keys = list(pv(MACHINES[0], MACHINES[0]).keys())
for j, m in enumerate(MACHINES):
    d = pv(m, MACHINES[0] if j else None)
    for k, v in d.items():
        L.append(r'\expandafter\def\csname pv@%s@%s\endcsname{%s}' % ('ABCDEFGH'[j], k, v))
if GPUS:
    for k, v in gpu_macros(GPUS[0], MACHINES[-1], MACHINES[0]).items():
        L.append(r'\expandafter\def\csname pv@G@%s\endcsname{%s}' % (k, v))
L.append(r'\def\pvnmachines{%d}' % len(MACHINES))
L.append(r'\def\pvngpus{%d}' % len(GPUS))
L.append(r'\makeatother')
open(os.path.join(OUT, 'parallel_macros.tex'), 'w').write('\n'.join(L) + '\n')
print('wrote tables/parallel_macros.tex (%d machines)' % len(MACHINES))
