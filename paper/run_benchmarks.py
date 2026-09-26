"""Benchmark this computer for Sec. VI G (Figs. 10, 11 and Table VII) and add it to the machine list of the paper.

    python run_benchmarks.py              full run: strong scaling (compute_speedup.py) and wall time versus T
                                          (compute_time_to_accuracy.py), then figures, tables and text macros
    python run_benchmarks.py --quick      2-minute smoke test on a 150^2 grid (not added to the paper)
    python run_benchmarks.py --only split  only the cost of each grid indicator alone (compute_readout_split.py)

Run it from an Anaconda prompt (numpy, numba, matplotlib; tbb recommended: `conda install tbb`, the reference
machine used Numba's TBB threading layer) with the computer otherwise idle and plugged in.  Close other programs:
background load changes the wall times.  The script detects the processor, the physical cores and the hardware
threads, chooses the thread counts (1-4, fractions of the physical cores, up to all hardware threads) and writes
    ../data/machines/<host>/speedup.json, speedup.log, time_to_accuracy.json, time_to_accuracy.log
and registers the machine in ../data/machines.json, which fig10_speedup.py, fig11_time_to_accuracy.py and
make_tables.py read.  Expected duration: about 1 h on a 6-core i5-10400; less on faster machines.

Options: --tag NAME (default: host name), --threads 1,2,4,... , --only speedup|tta|split, --no-plots, --label TEXT
(processor name used in the paper, default: from the processor string).
"""
import os, sys, json, time, argparse, io

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
sys.path.insert(0, HERE)
import machine_info


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s); st.flush()
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def logged(fn, logfile, *args):
    old = sys.stdout
    with open(logfile, 'w') as f:
        sys.stdout = Tee(old, f)
        try:
            t0 = time.time(); fn(*args); print(f'[finished in {(time.time() - t0)/60:.1f} min]')
        finally:
            sys.stdout = old


def register(entry):
    path = os.path.join(DATA, 'machines.json')
    M = json.load(open(path)) if os.path.exists(path) else []
    M = [m for m in M if m['tag'] != entry['tag']] + [entry]
    with open(path, 'w') as f:
        json.dump(M, f, indent=1)
    print('registered', entry['tag'], 'in', os.path.normpath(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag'); ap.add_argument('--threads'); ap.add_argument('--label')
    ap.add_argument('--only', choices=['speedup', 'tta', 'split']); ap.add_argument('--no-plots', action='store_true')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    MI = machine_info.info(); tag = a.tag or machine_info.tag_of(MI)
    nc, nl = MI['n_cores'], MI['n_logical']
    threads = a.threads or ','.join(map(str, machine_info.default_threads(nc, nl)))
    print('=' * 78)
    print(f"machine   {MI['cpu_name']}  ({nc} cores, {nl} hardware threads)")
    print(f"system    {MI['os']}, Python {MI['python']}, Numba {MI['numba']}")
    print(f"tag       {tag}      threads {threads}")
    print('=' * 78, flush=True)
    if MI['numba'] == 'not installed':
        sys.exit('numba is required: conda install numba tbb')
    try:
        known = json.load(open(os.path.join(DATA, 'machines.json')))
    except Exception:
        known = []
    for m in known:
        if m['cpu_name'] == MI['cpu_name'] and m['tag'] != tag and not a.quick:
            print(f"NOTE: same processor as the registered machine '{m['tag']}'.")

    import compute_speedup, compute_time_to_accuracy, compute_readout_split
    if a.quick:
        out = os.path.join(DATA, 'machines', tag + '_quick'); os.makedirs(out, exist_ok=True)
        logged(compute_speedup.main, os.path.join(out, 'speedup.log'),
               ['150', '--threads', f'1,{nl}', '--out', os.path.join(out, 'speedup.json')])
        print('smoke test done; results in', os.path.normpath(out), '(not used by the paper)')
        return

    out = os.path.join(DATA, 'machines', tag); os.makedirs(out, exist_ok=True)
    if a.only in (None, 'speedup'):
        logged(compute_speedup.main, os.path.join(out, 'speedup.log'),
               ['600', '--threads', threads, '--out', os.path.join(out, 'speedup.json')])
    if a.only in (None, 'tta'):
        logged(compute_time_to_accuracy.main, os.path.join(out, 'time_to_accuracy.log'),
               ['--threads', str(nl), '--out', os.path.join(out, 'time_to_accuracy.json')])
    if a.only in (None, 'split'):                  # cost of each grid indicator alone (1, cores, all threads)
        logged(compute_readout_split.main, os.path.join(out, 'readout_split.log'),
               ['--out', os.path.join(out, 'readout_split.json')])
    if os.path.exists(os.path.join(out, 'speedup.json')) and os.path.exists(os.path.join(out, 'time_to_accuracy.json')):
        register(dict(tag=tag, label=a.label or machine_info.short_name(MI['cpu_name']), cpu_name=MI['cpu_name'],
                      clock_ghz=machine_info.clock_ghz(MI['cpu_name']), n_cores=nc, n_logical=nl, os=MI['os'],
                      numba=MI['numba'], speedup=f'machines/{tag}/speedup.json',
                      time_to_accuracy=f'machines/{tag}/time_to_accuracy.json',
                      readout_split=f'machines/{tag}/readout_split.json', reference=False))
        if not a.no_plots:
            import subprocess
            for s in ('fig10_speedup', 'fig11_time_to_accuracy', 'make_tables'):
                subprocess.run([sys.executable, os.path.join(HERE, s + '.py')], check=True, cwd=HERE)
            print('\nDone.  Recompile main.tex (pdflatex main; bibtex main; pdflatex main x2).')


if __name__ == '__main__':
    main()
