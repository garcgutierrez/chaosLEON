"""Shared post-processing of the timing data of Sec. VI G (used by fig10, fig11 and make_tables).

Machines are listed in data/machines.json (written by run_benchmarks.py); each entry points to its speedup.json
(strong scaling at T = 200) and time_to_accuracy.json (AUROC versus T and wall time of the SALI map versus T).
The reference machine (i5-10400) keeps the original files data/speedup.json and data/time_to_accuracy.json.

MACHINES: list of dicts (reference first) with the registry fields plus
  SP (wall times of speedup.json), P (thread counts), pmax (all hardware threads), ACC (time_to_accuracy.json).
wall(name, T, p, m=REF): wall time (s) of the complete 600^2 map of indicator `name` at integration time T with p
  threads on machine m (p = 'max': all hardware threads).
  SALI: t_p(200) w(T)/w(200), where w(T) is the wall time measured on m with pmax threads at several T (the early exit
        makes it sublinear in T), interpolated linearly in T;
  grid maps and tangent LEON: fixed work per step, t_p(200) T/200;
  t_p(200) is the wall time of the strong-scaling run, so all times are consistent with Fig. 10.
t_star(name, A): first integration time at which AUROC(T) reaches A (linear interpolation in log T), or None.
  The AUROC does not depend on the machine; the values of the reference machine are used.
"""
import os, json, numpy as np
from style import DATA

KIND = {'SALI': 'sali', 'LEON': 'leon', 'S_h': 'grid', 'B_h': 'grid', 'OFLI2_h': 'grid', 'LEON_h': 'grid'}


def _load():
    path = os.path.join(DATA, 'machines.json')
    reg = json.load(open(path)) if os.path.exists(path) else [dict(
        tag='reference', label='reference machine', n_cores=6, n_logical=12, speedup='speedup.json',
        time_to_accuracy='time_to_accuracy.json', reference=True)]
    out = []
    for m in reg:
        fs, ft = os.path.join(DATA, m['speedup']), os.path.join(DATA, m['time_to_accuracy'])
        if not (os.path.exists(fs) and os.path.exists(ft)):
            continue
        m = dict(m); sp = json.load(open(fs)); acc = json.load(open(ft))
        m['SP'] = sp['times']; m['P'] = [int(p) for p in sp['threads']]; m['pmax'] = max(m['P'])
        m['ACC'] = acc
        m['wall_T'] = acc.get('wall', acc.get('wall12')); m['wall_threads'] = acc.get('wall_threads', 12)
        rs = os.path.join(DATA, m['readout_split']) if m.get('readout_split') else None
        m['RS'] = json.load(open(rs))['times'] if rs and os.path.exists(rs) else None
        m['repeat'] = sp.get('repeat', {}); m['layer'] = sp.get('threading_layer', '?'); m['os'] = sp.get('os', m.get('os', '?'))
        out.append(m)
    out.sort(key=lambda m: not m.get('reference', False))
    return out


def _load_gpus():
    """GPU runs of compute_gpu.py (data/gpu/<host>/gpu.json, CUDA backend), as pseudo-machines with one 'thread
    count' p = 'gpu' (fp64 times; the fp32 times are in m['fp32'])."""
    import glob
    out = []
    for f in sorted(glob.glob(os.path.join(DATA, 'gpu', '*', 'gpu.json'))):
        g = json.load(open(f))
        if g.get('backend') != 'cuda' or '600' not in g.get('fp64', {}):
            continue
        r = g['fp64']['600']; W = r['times']
        m = dict(tag=os.path.basename(os.path.dirname(f)), label=g['gpu']['name'].replace('NVIDIA ', ''), gpu=g['gpu'],
                 host=g['machine'], SP={k: {'gpu': W[k]} for k in ('sali', 'leon', 'grid', 'grid_adv')}, P=['gpu'],
                 pmax='gpu', wall_T=r['wall_sali'], raw=g, reference=False)
        out.append(m)
    return out


MACHINES = _load()
GPUS = _load_gpus()
REF = MACHINES[0]
ACC = REF['ACC']


def pp(m, p):
    return m['pmax'] if p == 'max' else (p if p == 'gpu' else int(p))


def wall(name, T, p, m=None):
    m = m or REF; p = pp(m, p); k = KIND[name]
    if k == 'sali':
        Tw = np.array(sorted(map(float, m['wall_T']))); tw = np.array([m['wall_T'][str(int(t))] for t in Tw])
        T = np.asarray(T, float)
        w = np.where(T < Tw[0], tw[0]*T/Tw[0], np.interp(T, Tw, tw))    # below T = 50: proportional to T
        return m['SP']['sali'][str(p)]*w/np.interp(200.0, Tw, tw)
    return m['SP'][k][str(p)]*np.asarray(T)/200.0


def speedup(m, key, p):
    return m['SP'][key]['1']/m['SP'][key][str(pp(m, p))]


def t_star(name, A):
    T = np.array(ACC['T'][name]); a = np.array(ACC['auroc'][name])
    i = np.nonzero(a >= A)[0]
    if i.size == 0:
        return None
    i = i[0]
    if i == 0:
        return float(T[0])
    lt = np.interp(A, [a[i-1], a[i]], np.log([T[i-1], T[i]]))
    return float(np.exp(lt))


def auroc_maxdiff(m):
    """Largest difference between the AUROC(T) of machine m and of the reference machine (reproducibility)."""
    d = 0.0
    for k, v in m['ACC']['auroc'].items():
        d = max(d, float(np.max(np.abs(np.array(v) - np.array(ACC['auroc'][k])))))
    return d
