"""GPU test for Sec. VI G: the complete Henon-Heiles maps of Fig. 4 computed on a graphics processor (CUDA, CuPy).

Same problem, integrator and indicators as compute_speedup.py (E = 1/8, section x = 0, N^2 grid in (y, p_y), T = 200,
Yoshida-4 with dt = 0.01, read-out every TAU = 500 steps).  One GPU thread per node:
  sali      SALI map (orbit + two deviation vectors, early exit once SALI < 1e-16)
  leon      tangent LEON map (orbit + exact discrete first..fourth variations along y)
  grid      grid-as-ensemble maps S_h, B_h, OFLI2_h, LEON_h: per block, one kernel advances every node TAU steps and a
            second kernel reads out the five- and nine-point stencils (the same block-synchronous scheme as on the CPU)
  grid_adv  the advance kernel alone
Precision: double (fp64, the reference) and single (fp32).  Consumer GPUs execute fp64 at 1/32-1/64 of the fp32 rate,
so fp32 is much faster; its accuracy is measured, not assumed.  Checks (N = 600): grid maps against data/fig4_maps.npz,
and the AUROC of every map on the labelled nodes of Fig. 4.  Larger grids (--sizes) are timed without checks.  The wall
time of the fp64 SALI map is also measured at the integration times of compute_time_to_accuracy.py (Fig. 11, Table VII).
The kernels are compiled with --fmad=false so that the orbits follow the same floating-point operations as on the CPU.

Usage:  python compute_gpu.py [--sizes 600,1200] [--no-fp32]     -> ../data/gpu/<host>/gpu.json and gpu.log
        (needs an NVIDIA GPU and CuPy:  conda install -c conda-forge cupy)
        python compute_gpu.py --backend emu --sizes 120 --no-wall      runs the same kernel source compiled as C++ with
        OpenMP on the CPU (needs g++), to test the kernels without a GPU
"""
import numpy as np, math, time, os, sys, json, argparse, ctypes, subprocess, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from compute_fig4_maps import E, T, DT, TAU, W1, W0, auc
from compute_speedup import neighbours, ic_jets
import machine_info

TWALL = [50, 100, 200, 300, 500, 700, 1000, 1500, 2000]

KSRC = r'''
#ifdef CPU_EMU
#include <cmath>
#define __global__
#define __device__
#define __restrict__
#define KLOOP(k, k0, k1) _Pragma("omp parallel for schedule(dynamic, 256)") for (long long k = (k0); k < (k1); ++k)
#else
#define KLOOP(k, k0, k1) long long k = (k0) + (long long)blockIdx.x*blockDim.x + threadIdx.x; if (k < (k1))
#endif
#if REAL_FLOAT
typedef float real;
#define SQRT sqrtf
#define LOG logf
#define POW powf
#define TINY 1e-37f
#define SALI_MIN 0.0f
#else
typedef double real;
#define SQRT sqrt
#define LOG log
#define POW pow
#define TINY 1e-300
#define SALI_MIN 1e-16
#endif
#define R(x) ((real)(x))
__device__ inline real rmax(real a, real b) { return (b > a) ? b : a; }    /* Python max(a, b) */

__device__ inline void kick(real* s, real h) {
    real x = s[0], y = s[1];
    s[2] -= h*(x + R(2)*x*y); s[3] -= h*(y + x*x - y*y);
}
__device__ inline void orbit(real* s, int n) {
    const real c3[3] = {R(W1_), R(W0_), R(W1_)};
    for (int it = 0; it < n; ++it)
        for (int q = 0; q < 3; ++q) {
            real h = c3[q]*R(DT_);
            kick(s, R(0.5)*h); s[0] += h*s[2]; s[1] += h*s[3]; kick(s, R(0.5)*h);
        }
}
__device__ inline real dot4(const real* a, const real* b) { real c = 0; for (int i = 0; i < 4; ++i) c += a[i]*b[i]; return c; }
/* section projection P: removes the grad H and flow directions */
__device__ inline void proj(const real* s, const real* v, real* out) {
    real x = s[0], y = s[1], px = s[2], py = s[3];
    real g[4] = {x + R(2)*x*y, y + x*x - y*y, px, py};
    real ng = SQRT(dot4(g, g)); for (int i = 0; i < 4; ++i) g[i] /= ng;
    real f[4] = {px, py, -(x + R(2)*x*y), -(y + x*x - y*y)};
    real fg = dot4(f, g); for (int i = 0; i < 4; ++i) f[i] -= fg*g[i];
    real nf = SQRT(dot4(f, f)); for (int i = 0; i < 4; ++i) f[i] /= nf;
    real vg = dot4(v, g), vf = dot4(v, f);
    for (int i = 0; i < 4; ++i) out[i] = v[i] - vg*g[i] - vf*f[i];
}
__device__ inline real wedge(const real* a, const real* b) {
    real na2 = dot4(a, a), ab = dot4(a, b)/na2, bp[4];
    for (int i = 0; i < 4; ++i) bp[i] = b[i] - ab*a[i];
    return SQRT(na2*dot4(bp, bp));
}

extern "C" __global__ void advance(real* __restrict__ S, long long n, long long k0, long long k1, int nsteps) {
    KLOOP(k, k0, k1) {
        real s[4]; for (int c = 0; c < 4; ++c) s[c] = S[c*n + k];
        orbit(s, nsteps);
        for (int c = 0; c < 4; ++c) S[c*n + k] = s[c];
    }
}

extern "C" __global__ void readout(const real* __restrict__ S, const long long* __restrict__ NB, long long n,
                                   long long k0, long long k1, real hy, real hp, real* __restrict__ F, int mode) {
    /* mode -1: all four indicators; 0 S_h, 1 B_h, 2 OFLI2_h, 3 LEON_h alone */
    KLOOP(k, k0, k1) {
        real x0[4], F0 = F[k], F1 = F[n + k], F2 = F[2*n + k], F3 = F[3*n + k];
        for (int c = 0; c < 4; ++c) x0[c] = S[c*n + k];
        for (int d = 0; d < 2; ++d) {
            real h = d == 0 ? hy : hp;
            long long i1 = NB[k*8 + d*4], j1 = NB[k*8 + d*4 + 1], i2 = NB[k*8 + d*4 + 2], j2 = NB[k*8 + d*4 + 3];
            if (i1 < 0 || j1 < 0) continue;
            bool has2 = i2 >= 0 && j2 >= 0;
            real xp1[4], xm1[4], xp2[4], xm2[4], D[4], C[4], pD[4], pC[4], t[4], o[4];
            for (int c = 0; c < 4; ++c) {
                xp1[c] = S[c*n + i1]; xm1[c] = S[c*n + j1];
                xp2[c] = has2 ? S[c*n + i2] : xp1[c]; xm2[c] = has2 ? S[c*n + j2] : xm1[c];
                D[c] = (xp1[c] - xm1[c])/(R(2)*h); C[c] = (xp1[c] + xm1[c] - R(2)*x0[c])/(h*h);
            }
            if (mode < 0 || mode == 0 || mode == 1) {
                proj(x0, D, pD); real nD = SQRT(dot4(pD, pD));
                if (mode != 1) F0 = rmax(F0, LOG(nD));
                if (mode != 0) { proj(x0, C, pC); F1 = rmax(F1, LOG(rmax(h*wedge(pD, pC)/(nD*nD), TINY))); }
            }
            if (mode < 0 || mode == 2) {
                for (int c = 0; c < 4; ++c) t[c] = D[c] + R(0.5)*C[c];
                proj(x0, t, o); F2 = rmax(F2, LOG(SQRT(dot4(o, o))));
            }
            if (has2 && (mode < 0 || mode == 3)) {
                real c1[4], c4[4];
                for (int c = 0; c < 4; ++c) t[c] = (R(8)*(xp1[c] - xm1[c]) - (xp2[c] - xm2[c]))/R(12);
                proj(x0, t, c1);
                for (int c = 0; c < 4; ++c) t[c] = (xp2[c] - R(4)*xp1[c] + R(6)*x0[c] - R(4)*xm1[c] + xm2[c])/R(24);
                proj(x0, t, c4);
                F3 = rmax(F3, LOG(rmax(wedge(c1, c4)/dot4(c1, c1), TINY))/R(3));
            }
        }
        F[k] = F0; F[n + k] = F1; F[2*n + k] = F2; F[3*n + k] = F3;
    }
}

__device__ inline void tangent(real* s, real V[4][2], int n) {
    const real c3[3] = {R(W1_), R(W0_), R(W1_)};
    for (int it = 0; it < n; ++it)
        for (int q = 0; q < 3; ++q) {
            real h = c3[q]*R(DT_);
            for (int half = 0; half < 2; ++half) {
                real A = R(1) + R(2)*s[1], B = R(2)*s[0], D = R(1) - R(2)*s[1];
                for (int j = 0; j < 2; ++j) {
                    V[2][j] -= R(0.5)*h*(A*V[0][j] + B*V[1][j]); V[3][j] -= R(0.5)*h*(B*V[0][j] + D*V[1][j]);
                }
                kick(s, R(0.5)*h);
                if (half == 0) {
                    s[0] += h*s[2]; s[1] += h*s[3];
                    for (int j = 0; j < 2; ++j) { V[0][j] += h*V[2][j]; V[1][j] += h*V[3][j]; }
                }
            }
        }
}

/* SALI map, one block of TAU steps per launch (short kernels: no display-driver time-out); state in global memory:
   S (4, n) orbit, V (8, n) deviation vectors (V[i][j] at (2i + j) n), val (n) current SALI */
extern "C" __global__ void sali_block(real* __restrict__ S, real* __restrict__ Vg, real* __restrict__ valg, long long n,
                                      long long k0, long long k1) {
    KLOOP(k, k0, k1) {
        if (!(valg[k] < SALI_MIN)) {
            real s[4], V[4][2];
            for (int c = 0; c < 4; ++c) s[c] = S[c*n + k];
            for (int i = 0; i < 4; ++i) for (int j = 0; j < 2; ++j) V[i][j] = Vg[(2*i + j)*n + k];
            tangent(s, V, TAU_);
            real na = 0, nc = 0;
            for (int i = 0; i < 4; ++i) { na += V[i][0]*V[i][0]; nc += V[i][1]*V[i][1]; }
            na = SQRT(na); nc = SQRT(nc);
            real sp = 0, sm = 0;
            for (int i = 0; i < 4; ++i) {
                V[i][0] /= na; V[i][1] /= nc;
                sp += (V[i][0] + V[i][1])*(V[i][0] + V[i][1]); sm += (V[i][0] - V[i][1])*(V[i][0] - V[i][1]);
            }
            sp = SQRT(sp); sm = SQRT(sm); valg[k] = sp < sm ? sp : sm;
            for (int c = 0; c < 4; ++c) S[c*n + k] = s[c];
            for (int i = 0; i < 4; ++i) for (int j = 0; j < 2; ++j) Vg[(2*i + j)*n + k] = V[i][j];
        }
    }
}

/* tangent LEON: exact discrete first..fourth variations Z[0..3] of the Yoshida-4 leapfrog (cubic potential) */
__device__ inline void hh_df(real x, real y, const real* a, real* r) {
    r[0] = -(R(1) + R(2)*y)*a[0] - R(2)*x*a[1]; r[1] = -R(2)*x*a[0] - (R(1) - R(2)*y)*a[1];
}
__device__ inline void hh_d2(const real* a, const real* b, real* r) {
    r[0] = -R(2)*(a[0]*b[1] + a[1]*b[0]); r[1] = -R(2)*a[0]*b[0] + R(2)*a[1]*b[1];
}
__device__ inline void jet_step(real* s, real Z[4][4], int n) {
    const real c3[3] = {R(W1_), R(W0_), R(W1_)};
    for (int it = 0; it < n; ++it)
        for (int q = 0; q < 3; ++q) {
            real h = c3[q]*R(DT_);
            for (int half = 0; half < 2; ++half) {
                real hk = R(0.5)*h, x = s[0], y = s[1], a[2], b[2], b2[2], g[2], g2[2], e[2], e2[2], e3[2];
                hh_df(x, y, Z[0], a); hh_df(x, y, Z[1], b); hh_d2(Z[0], Z[0], b2);
                hh_df(x, y, Z[2], g); hh_d2(Z[0], Z[1], g2);
                hh_df(x, y, Z[3], e); hh_d2(Z[0], Z[2], e2); hh_d2(Z[1], Z[1], e3);
                for (int i = 0; i < 2; ++i) {
                    Z[0][2+i] += hk*a[i]; Z[1][2+i] += hk*(b[i] + b2[i]); Z[2][2+i] += hk*(g[i] + R(3)*g2[i]);
                    Z[3][2+i] += hk*(e[i] + R(4)*e2[i] + R(3)*e3[i]);
                }
                kick(s, hk);
                if (half == 0) {
                    s[0] += h*s[2]; s[1] += h*s[3];
                    for (int r = 0; r < 4; ++r) { Z[r][0] += h*Z[r][2]; Z[r][1] += h*Z[r][3]; }
                }
            }
        }
}

/* tangent LEON map, one block per launch; state: S (4, n), Z (16, n) (Z[r][c] at (4r + c) n), Lc, sup (n) */
extern "C" __global__ void leon_block(real* __restrict__ S, real* __restrict__ Zg, real* __restrict__ Lcg,
                                      real* __restrict__ supg, long long n, long long k0, long long k1) {
    KLOOP(k, k0, k1) {
        real s[4], Z[4][4], p0[4], p3[4], Lc = Lcg[k], sup = supg[k];
        for (int c = 0; c < 4; ++c) s[c] = S[c*n + k];
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) Z[r][c] = Zg[(4*r + c)*n + k];
        jet_step(s, Z, TAU_);
        proj(s, Z[0], p0); proj(s, Z[3], p3); real n1 = SQRT(dot4(p0, p0));
        sup = rmax(sup, Lc + (LOG(rmax(wedge(p0, p3), TINY)) - R(2)*LOG(n1) - R(LOG24_))/R(3));
        real c = SQRT(dot4(Z[0], Z[0])); Lc += LOG(c);
        for (int r = 0; r < 4; ++r) { real cr = POW(c, R(r + 1)); for (int i = 0; i < 4; ++i) Z[r][i] /= cr; }
        for (int c2 = 0; c2 < 4; ++c2) S[c2*n + k] = s[c2];
        for (int r = 0; r < 4; ++r) for (int c2 = 0; c2 < 4; ++c2) Zg[(4*r + c2)*n + k] = Z[r][c2];
        Lcg[k] = Lc; supg[k] = sup;
    }
}
'''
DEFS = {'W1_': repr(W1), 'W0_': repr(W0), 'DT_': repr(DT), 'TAU_': str(TAU), 'LOG24_': repr(math.log(24.0))}


def source(fp32):
    head = f'#define REAL_FLOAT {1 if fp32 else 0}\n' + ''.join(f'#define {k} {v}\n' for k, v in DEFS.items())
    return head + KSRC


# ----------------------------------------------------------------------------- backends
class Cuda:
    name = 'cuda'

    def __init__(self, fp32, block=128):
        import cupy as cp
        self.cp = cp; self.block = block; self.dtype = np.float32 if fp32 else np.float64
        self.mod = cp.RawModule(code=source(fp32), options=('--fmad=false', '-std=c++14'))
        self.f = {k: self.mod.get_function(k) for k in ('advance', 'readout', 'sali_block', 'leon_block')}

    def dev(self, a):
        return self.cp.asarray(a)

    def host(self, a):
        return self.cp.asnumpy(a)

    def sync(self):
        self.cp.cuda.Device().synchronize()

    def launch(self, name, n, *args):
        self.f[name](((n + self.block - 1)//self.block,), (self.block,), args)

    @staticmethod
    def info():
        import cupy as cp
        p = cp.cuda.runtime.getDeviceProperties(0)
        nm = p['name'].decode() if isinstance(p['name'], bytes) else str(p['name'])
        return dict(name=nm, compute_capability=f"{p['major']}.{p['minor']}", multiprocessors=int(p['multiProcessorCount']),
                    memory_GB=round(p['totalGlobalMem']/2**30, 1), clock_MHz=int(p.get('clockRate', 0))//1000,
                    driver=cp.cuda.runtime.driverGetVersion(), runtime=cp.cuda.runtime.runtimeGetVersion(),
                    cupy=cp.__version__)


class Emu:
    """The kernel source compiled as C++/OpenMP on the CPU (validation of the kernels without a GPU)."""
    name = 'emu'

    def __init__(self, fp32):
        self.dtype = np.float32 if fp32 else np.float64
        d = tempfile.mkdtemp(); src = os.path.join(d, 'k.cpp'); lib = os.path.join(d, 'k.so')
        open(src, 'w').write(source(fp32))
        subprocess.run(['g++', '-O2', '-fopenmp', '-ffp-contract=off', '-shared', '-fPIC', '-DCPU_EMU', src, '-o', lib],
                       check=True)
        self.lib = ctypes.CDLL(lib)

    def dev(self, a):
        return np.ascontiguousarray(a)

    host = dev

    def sync(self):
        pass

    def launch(self, name, n, *args):
        conv = []
        for a in args:
            if isinstance(a, np.ndarray):
                conv.append(a.ctypes.data_as(ctypes.c_void_p))
            elif isinstance(a, np.int64):
                conv.append(ctypes.c_longlong(int(a)))
            elif isinstance(a, np.int32):
                conv.append(ctypes.c_int(int(a)))
            elif isinstance(a, np.float32):
                conv.append(ctypes.c_float(float(a)))
            else:
                conv.append(ctypes.c_double(float(a)))
        getattr(self.lib, name)(*conv)

    @staticmethod
    def info():
        return dict(name='CPU emulation (g++/OpenMP)', compute_capability='-')


# ----------------------------------------------------------------------------- problem and maps
def problem(N):
    yv = np.linspace(-0.6, 0.8, N); pv = np.linspace(-0.6, 0.6, N); hy, hp = yv[1] - yv[0], pv[1] - pv[0]
    Y, P = np.meshgrid(yv, pv, indexing='ij'); v = 2*E - P**2 - Y**2 + 2*Y**3/3
    PX = np.sqrt(np.maximum(v, 0)); ok = (v > 0) & (PX >= 1e-3)
    Sg = np.zeros((N, N, 4)); Sg[..., 1] = Y; Sg[..., 2] = PX; Sg[..., 3] = P
    idx, NB = neighbours(ok); S0 = Sg[ok].copy()
    return dict(N=N, hy=hy, hp=hp, ok=ok, idx=idx, NB=np.ascontiguousarray(NB, dtype=np.int64), S0=S0, n=len(S0))


CHUNK = 1 << 17          # nodes per kernel launch: keeps every launch short (Windows display-driver time-out)


class Maps:
    def __init__(self, be, pb):
        self.be = be; self.pb = pb; dt = be.dtype; n = pb['n']
        self.S0 = be.dev(np.ascontiguousarray(pb['S0'].T.astype(dt)))           # SoA (4, n)
        self.NB = be.dev(pb['NB'])
        Z = ic_jets(pb['S0'][:, 1], pb['S0'][:, 3]).reshape(n, 16)              # (n, 16) -> SoA (16, n)
        self.Z0 = be.dev(np.ascontiguousarray(Z.T.astype(dt)))
        self.n = np.int64(n); self.hy = dt(pb['hy']); self.hp = dt(pb['hp'])
        self.chunks = [(np.int64(a), np.int64(min(a + CHUNK, n))) for a in range(0, n, CHUNK)]

    def timed(self, f, *a):
        self.be.sync(); t0 = time.perf_counter(); r = f(*a); self.be.sync(); return time.perf_counter() - t0, r

    def grid(self, nblk, readout=True, mode=-1):
        be, n = self.be, int(self.n)
        S = self.S0.copy(); F = be.dev(np.full((4, n), -np.inf, dtype=be.dtype))
        for b in range(nblk):
            for k0, k1 in self.chunks:
                be.launch('advance', int(k1 - k0), S, self.n, k0, k1, np.int32(TAU))
            if readout:
                for k0, k1 in self.chunks:
                    be.launch('readout', int(k1 - k0), S, self.NB, self.n, k0, k1, self.hy, self.hp, F, np.int32(mode))
        return F

    def sali(self, nblk):
        be, n, dt = self.be, int(self.n), self.be.dtype
        S = self.S0.copy(); V = np.zeros((8, n), dtype=dt); V[0] = 1; V[5] = 1      # V[0][0] = 1, V[2][1] = 1
        V = be.dev(V); val = be.dev(np.ones(n, dtype=dt))
        for b in range(nblk):
            for k0, k1 in self.chunks:
                be.launch('sali_block', int(k1 - k0), S, V, val, self.n, k0, k1)
        return val

    def leon(self, nblk):
        be, n, dt = self.be, int(self.n), self.be.dtype
        S = self.S0.copy(); Z = self.Z0.copy(); Lc = be.dev(np.zeros(n, dtype=dt))
        sup = be.dev(np.full(n, -np.inf, dtype=dt))
        for b in range(nblk):
            for k0, k1 in self.chunks:
                be.launch('leon_block', int(k1 - k0), S, Z, Lc, sup, self.n, k0, k1)
        return sup


def labels(pb):
    ref = np.load(os.path.join(HERE, '..', 'data', 'fig4_maps.npz'))
    lab = np.where(ref['ref'] < 1e-8, 1, np.where(ref['ref'] > 1e-4, 0, -1)); kk = lab >= 0
    return ref, lab[kk], ref['ref_ij'][kk, 0], ref['ref_ij'][kk, 1]


def to_map(pb, v):
    m = np.full(pb['ok'].shape, np.nan); m[pb['ok']] = v; m[~np.isfinite(m)] = np.nan; return m


def score(pb, lab, ii, jj, v):
    z = to_map(pb, v)[ii, jj]; g = np.isfinite(z); return float(auc(lab[g], z[g]))


def run_precision(Backend, fp32, sizes, res, do_wall):
    tag = 'fp32' if fp32 else 'fp64'; be = Backend(fp32); out = res[tag] = {}
    nblk = int(round(T/(TAU*DT)))
    for N in sizes:
        pb = problem(N); mp = Maps(be, pb); r = out[str(N)] = dict(n_nodes=int(pb['n']), times={})
        mp.grid(1); mp.sali(1); mp.leon(1); be.sync()                                   # warm-up
        W = r['times']
        W['grid'], F = mp.timed(mp.grid, nblk, True)
        W['grid_adv'], _ = mp.timed(mp.grid, nblk, False)
        Fall = be.host(F)
        for m, name in enumerate(('S_h', 'B_h', 'OFLI2_h', 'LEON_h')):            # one grid indicator at a time
            W['grid_' + name], Fm = mp.timed(mp.grid, nblk, True, m)
            a, b = Fall[m], be.host(Fm)[m]; g = np.isfinite(a) & np.isfinite(b)
            r.setdefault('check_single', {})[name] = [float(np.max(np.abs(a[g] - b[g]))), int(np.sum(np.isfinite(a) != np.isfinite(b)))]
        W['sali'], Sm = mp.timed(mp.sali, nblk)
        W['leon'], Lm = mp.timed(mp.leon, nblk)
        W['grid_repeat'], _ = mp.timed(mp.grid, nblk, True); W['sali_repeat'], _ = mp.timed(mp.sali, nblk)
        print(f'{tag} N={N} ({pb["n"]} nodes): ' + ', '.join(f'{k} {v:.2f} s' for k, v in W.items()), flush=True)
        if N == 600:
            ref, lab, ii, jj = labels(pb); F = be.host(F).astype(np.float64); c = r['check'] = {}; a = r['auroc'] = {}
            for j, name in enumerate(('S_h', 'B_h', 'OFLI2_h', 'LEON_h')):
                m = to_map(pb, F[j]); rr = ref[name]; both = np.isfinite(m) & np.isfinite(rr); d = np.abs(m[both] - rr[both])
                c[name + '_maxdiff'] = float(d.max()); c[name + '_frac_gt_1e-6'] = float(np.mean(d > 1e-6))
                c[name + '_nan_mismatch'] = int(np.sum(np.isfinite(m) != np.isfinite(rr)))
                a[name] = score(pb, lab, ii, jj, F[j])
            sv = be.host(Sm).astype(np.float64)
            a['SALI'] = score(pb, lab, ii, jj, -np.log10(np.maximum(sv, 1e-16)))
            a['LEON'] = score(pb, lab, ii, jj, be.host(Lm).astype(np.float64))
            a['reference_cpu'] = dict(zip([str(s) for s in ref['auc_names']], map(float, ref['auc_vals'])))
            print(f'{tag} checks', c, flush=True); print(f'{tag} AUROC', {k: round(v, 4) for k, v in a.items() if k != 'reference_cpu'},
                                                           'CPU:', {k: round(v, 4) for k, v in a['reference_cpu'].items()}, flush=True)
        if N == 600 and do_wall and not fp32:
            r['wall_sali'] = {}
            for Tw in TWALL:
                t, _ = mp.timed(mp.sali, int(round(Tw/(TAU*DT)))); r['wall_sali'][Tw] = t
                print(f'{tag} SALI map T={Tw}: {t:.2f} s', flush=True)
        save(res)


def save(res):
    with open(res['_out'], 'w') as f:
        json.dump({k: v for k, v in res.items() if k != '_out'}, f, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['cuda', 'emu'], default='cuda')
    ap.add_argument('--sizes', default='600,1200'); ap.add_argument('--no-fp32', action='store_true')
    ap.add_argument('--no-wall', action='store_true'); ap.add_argument('--tag'); ap.add_argument('--out')
    a = ap.parse_args(argv)
    Backend = Cuda if a.backend == 'cuda' else Emu
    try:
        gi = Backend.info()
    except Exception as e:
        sys.exit(f'No usable CUDA GPU / CuPy ({e!r}).  Install CuPy (conda install -c conda-forge cupy) or use --backend emu.')
    MI = machine_info.info(); tag = a.tag or machine_info.tag_of(MI)
    outdir = os.path.join(HERE, '..', 'data', 'gpu', tag + ('' if a.backend == 'cuda' else '_emu')); os.makedirs(outdir, exist_ok=True)
    res = dict(_out=a.out or os.path.join(outdir, 'gpu.json'), backend=a.backend, gpu=gi, machine=MI, T=T, dt=DT,
               tau_steps=TAU, block=128)
    sys.stdout = Tee(sys.__stdout__, open(res['_out'].replace('.json', '.log'), 'w'))
    print('GPU', gi, '\nhost', MI, flush=True)
    sizes = [int(s) for s in a.sizes.split(',')]
    run_precision(Backend, False, sizes, res, not a.no_wall)
    if not a.no_fp32:
        run_precision(Backend, True, sizes, res, False)
    print('wrote', os.path.normpath(res['_out']))


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        for st in self.s:
            st.write(x); st.flush()
        return len(x)

    def flush(self):
        for st in self.s:
            st.flush()


if __name__ == '__main__':
    main()
