"""Usage: python grid_nodes_hh.py 1000 2000 out.npz 0.125   (grid N, sampled nodes, output, energy)
Grid-as-ensemble chaos indicators for the Henon-Heiles PSS (x=0).
Output F[..., k]: 0 stretch, 1 kernel SALI (aspect ratio), 2 fold B_h, 3 grid-OFLI2 |P(D+S/2)|, 4 finite-size |P(D+hS/2)|.

Idea: on a full grid every initial condition is also a sample of its neighbours'
uncertainty cloud ("overlapping sigma" with sigma ~ grid spacing h).  The flow map
evaluated on the grid therefore already contains the finite-size deformation of the
probability density:
    first differences  D_y = [x(y+h) - x(y-h)]/2h           (finite-size Jacobian columns)
    second differences S_y = [x(y+h) + x(y-h) - 2x(y)]/h^2   (finite-size curvature / folding)
so chaos indicators cost ONE trajectory per grid point (no variational equations).
Here we emulate a full grid by integrating each sampled node together with its
4 grid neighbours (in a real full-grid run those neighbours are other grid nodes).
Features per horizon (all with section projection P at the centre state):
  fd_ofli : sup_t log max(|P D_y|, |P D_p|)
  fd_sali : SALI(P D_y, P D_p)
  fd_fold : sup_t max_dir  h |P D ^ P S| / |P D|^2     (finite-size bending angle)
"""
import numpy as np, numba as nb, math, sys, time
from new_indicators_hh import W1, W0, proj, sali_pair, sali_ref, px_from   # same folder
from fold_indicator_hh import wedge                                           # same folder

@nb.njit(cache=True, inline='always')
def lf(s, h):
    x, y, px, py = s[0], s[1], s[2], s[3]
    px -= 0.5*h*(x + 2*x*y); py -= 0.5*h*(y + x*x - y*y)
    x += h*px; y += h*py
    px -= 0.5*h*(x + 2*x*y); py -= 0.5*h*(y + x*x - y*y)
    s[0] = x; s[1] = y; s[2] = px; s[3] = py

@nb.njit(cache=True)
def y4n(S, dt, n):
    for _ in range(n):
        for c in (W1, W0, W1):
            for k in range(S.shape[0]):
                lf(S[k], c*dt)

@nb.njit(cache=True)
def run_node(S0, hy, hp, dt, tau, nbk, hidx):
    """S0: (5,4) states: centre, y+, y-, p+, p-."""
    S = S0.copy(); out = np.full((hidx.size, 5), np.nan)
    so = -1e300; sf = -1e300; s2 = -1e300; s2h = -1e300; hk = 0
    for b in range(nbk):
        y4n(S, dt, tau)
        c = S[0]
        Dy = (S[1] - S[2])/(2*hy); Dp = (S[3] - S[4])/(2*hp)
        Sy = (S[1] + S[2] - 2*c)/(hy*hy); Sp = (S[3] + S[4] - 2*c)/(hp*hp)
        pDy = proj(c, Dy); pDp = proj(c, Dp); pSy = proj(c, Sy); pSp = proj(c, Sp)
        ny = math.sqrt(np.sum(pDy*pDy)); npp = math.sqrt(np.sum(pDp*pDp))
        so = max(so, math.log(max(ny, npp)))
        by = hy*wedge(pDy, pSy)/(ny*ny); bp = hp*wedge(pDp, pSp)/(npp*npp)
        sf = max(sf, math.log(max(by, bp, 1e-300)))
        # grid OFLI2: |P(D + S/2)| (Barrio's combination with unit segment) and finite-size |P(D + h S/2)|
        o2y = proj(c, Dy + 0.5*Sy); o2p = proj(c, Dp + 0.5*Sp)
        s2 = max(s2, math.log(max(math.sqrt(np.sum(o2y*o2y)), math.sqrt(np.sum(o2p*o2p)))))
        fy = proj(c, Dy + 0.5*hy*Sy); fp = proj(c, Dp + 0.5*hp*Sp)
        s2h = max(s2h, math.log(max(math.sqrt(np.sum(fy*fy)), math.sqrt(np.sum(fp*fp)))))
        if hk < hidx.size and b == hidx[hk]:
            out[hk, 0] = so; out[hk, 1] = sali_pair(pDy, pDp); out[hk, 2] = sf; out[hk, 3] = s2; out[hk, 4] = s2h; hk += 1
    return out

@nb.njit(parallel=True, cache=True)
def run_nodes(S0all, hy, hp, dt, tau, nbk, hidx):
    out = np.empty((S0all.shape[0], hidx.size, 5)); ref = np.empty(S0all.shape[0])
    for i in nb.prange(S0all.shape[0]):
        out[i] = run_node(S0all[i], hy, hp, dt, tau, nbk, hidx)
        ref[i] = sali_ref(S0all[i, 0], dt, tau, 2000)
    return out, ref

def sample(Ngrid, nsamp, seed=5, E=0.125):
    yv = np.linspace(-0.6, 0.8, Ngrid); pv = np.linspace(-0.6, 0.6, Ngrid)
    hy, hp = yv[1]-yv[0], pv[1]-pv[0]
    rng = np.random.default_rng(seed); S0 = []
    while len(S0) < nsamp:
        i, j = rng.integers(1, Ngrid-1, 2); y, py = yv[i], pv[j]
        st = []
        for yy, pp in ((y, py), (y+hy, py), (y-hy, py), (y, py+hp), (y, py-hp)):
            px = px_from(yy, pp, E)
            if not np.isfinite(px) or px < 1e-3: break
            st.append((0., yy, px, pp))
        if len(st) == 5: S0.append(st)
    return np.array(S0), hy, hp

if __name__ == '__main__':
    Ngrid = int(sys.argv[1]); nsamp = int(sys.argv[2]); fn = sys.argv[3]
    H = np.array([25, 50, 100, 150, 200, 300, 500]); hidx = (H/5 - 1).astype(np.int64)
    E = float(sys.argv[4]) if len(sys.argv) > 4 else 0.125
    S0, hy, hp = sample(Ngrid, nsamp, E=E)
    t0 = time.time(); F, ref = run_nodes(S0, hy, hp, 0.01, 500, int(H[-1]/5), hidx)
    print(fn, 'grid', Ngrid, 'h=', round(hy, 5), round(hp, 5), 'time', round(time.time()-t0, 1), flush=True)
    np.savez(fn, S0=S0, F=F, ref=ref, H=H, hy=hy, hp=hp)
