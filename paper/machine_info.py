"""Hardware description of the machine running the benchmarks (used by run_benchmarks.py and compute_*.py).

info() -> dict(hostname, cpu_name, n_cores, n_logical, os, python, numba)
  cpu_name   marketing name of the processor (Windows registry, /proc/cpuinfo or sysctl)
  n_cores    physical cores (psutil, else PowerShell/wmic on Windows, lscpu on Linux, sysctl on macOS)
  n_logical  hardware threads (os.cpu_count())
default_threads(n_cores, n_logical): thread counts for the strong-scaling study: 1-4, fractions of the physical
  cores, and points between one thread per core and all hardware threads.
Only the standard library is required (psutil is used when installed).
"""
import os, sys, re, platform, subprocess


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return ''


def cpu_name():
    s = platform.system()
    if s == 'Windows':
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
            return winreg.QueryValueEx(k, 'ProcessorNameString')[0].strip()
        except Exception:
            pass
    elif s == 'Linux':
        try:
            for line in open('/proc/cpuinfo'):
                if line.lower().startswith('model name'):
                    return line.split(':', 1)[1].strip()
        except Exception:
            pass
    elif s == 'Darwin':
        out = _run(['sysctl', '-n', 'machdep.cpu.brand_string']).strip()
        if out:
            return out
    return platform.processor() or 'unknown'


def physical_cores():
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    s = platform.system()
    if s == 'Windows':
        out = _run(['powershell', '-NoProfile', '-Command',
                    '(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum'])
        m = re.search(r'\d+', out)
        if m:
            return int(m.group())
        out = _run(['wmic', 'cpu', 'get', 'NumberOfCores'])
        v = [int(x) for x in re.findall(r'\d+', out)]
        if v:
            return sum(v)
    elif s == 'Linux':
        out = _run(['lscpu', '-p=CORE,SOCKET'])
        pairs = {l for l in out.splitlines() if l and not l.startswith('#')}
        if pairs:
            return len(pairs)
    elif s == 'Darwin':
        out = _run(['sysctl', '-n', 'hw.physicalcpu']).strip()
        if out.isdigit():
            return int(out)
    return os.cpu_count() or 1


def info():
    try:
        import numba
        nbv = numba.__version__
    except Exception:
        nbv = 'not installed'
    return dict(hostname=platform.node(), cpu_name=cpu_name(), n_cores=physical_cores(), n_logical=os.cpu_count() or 1,
                os=platform.platform(), python=platform.python_version(), numba=nbv)


def default_threads(nc, nl):
    s = {1, 2, 3, 4, nc, nl}
    s |= {max(1, round(nc*f)) for f in (0.5, 0.75)}
    if nl > nc:
        s |= {round(nc + (nl - nc)*f) for f in (1/3, 2/3)}
    return sorted(p for p in s if 1 <= p <= nl)


def short_name(name):
    """'Intel(R) Core(TM) i7-12700 CPU @ 2.10GHz' -> 'Intel Core i7-12700'; 'AMD Ryzen 9 7950X 16-Core Processor' -> 'AMD Ryzen 9 7950X'."""
    s = re.sub(r'\((R|TM|tm|r)\)', '', name)
    s = re.sub(r'^\s*\d+(st|nd|rd|th) Gen\s+', '', s)                  # '13th Gen Intel Core i9-13900H'
    s = re.sub(r'@.*$', '', s)
    s = re.sub(r'\b\d+-Core\b', '', s)
    s = re.sub(r'\b(CPU|Processor|with Radeon.*|w/ Radeon.*)\b', '', s)
    return re.sub(r'\s+', ' ', s).strip() or name


def clock_ghz(name):
    m = re.search(r'@\s*([\d.]+)\s*GHz', name)
    return float(m.group(1)) if m else None


def tag_of(inf):
    """Short file-system friendly name of the machine: host name, lower case."""
    t = re.sub(r'[^A-Za-z0-9_-]+', '-', inf['hostname'] or 'machine').strip('-').lower()
    return t or 'machine'


if __name__ == '__main__':
    i = info(); print(i); print('threads', default_threads(i['n_cores'], i['n_logical'])); print('tag', tag_of(i))
