"""Data for Fig. 1: covariance aspect ratio versus SALI along the Henon-Heiles orbits R, S, C (E = 1/8).

Symplectic Euler scheme, dt = 1e-3, sigma = 1e-2, trace renormalization every 2 time units (as in the paper).
For each orbit it records, every 2 time units:
  AR_sqrt   aspect ratio s2/s1 of the square-root factor L = Phi L0 (L0 = sigma [e_x, e_y]), raw (unnormalized)
  AR_expl   aspect ratio sqrt(lam2/lam1) of the explicitly updated covariance C <- C + dt (J C + C J^T)
            (eigenvalues clipped at zero, as a naive implementation would do)
  AR_hat    aspect ratio of the normalized directions, and SALI of the same pair (for panel b)
  SALI      standard SALI of the deviation vectors e_x, e_px
  ratio     ||w2|| / ||w1|| of the columns of L (unequal growth along regular orbits)
Usage:  python compute_fig1_aspect_ratio.py            -> ../data/fig1_aspect_ratio.npz
"""
import numpy as np, numba as nb, os

E = 0.125; DT = 1e-3; BLOCK = 2000; NBLK = 1000; SIG = 1e-2          # 2000 time units
ORBITS = {'R': 0.1, 'S': -0.0135, 'C': -0.015}                       # y0 (x0 = 0, py0 = 0, px0 > 0 from H = E)


@nb.njit(cache=True)
def jac(x, y):
    """d(force)/dq for H = (p^2 + q^2)/2 + x^2 y - y^3/3."""
    return -(1 + 2*y), -2*x, -(1 - 2*y)


@nb.njit(cache=True)
def run(s0, nblk, block, dt, sig):
    s = s0.copy()
    L = np.zeros((4, 2)); L[0, 0] = sig; L[1, 1] = sig                # square-root factor (x and y directions)
    N = L.copy()                                                     # normalized pair for AR_hat / SALI_pair
    W = np.zeros((4, 2)); W[0, 0] = 1.0; W[2, 1] = 1.0               # SALI pair e_x, e_px
    C = L @ L.T
    out = np.full((nblk, 7), np.nan)
    for b in range(nblk):
        for _ in range(block):
            x, y = s[0], s[1]
            a, bb, d = jac(x, y)
            # symplectic Euler: p' = p + dt F(q); q' = q + dt p'
            s[2] += dt*(-(x + 2*x*y)); s[3] += dt*(-(y + x*x - y*y))
            s[0] += dt*s[2]; s[1] += dt*s[3]
            for M in (L, N, W):
                for j in range(2):
                    M[2, j] += dt*(a*M[0, j] + bb*M[1, j]); M[3, j] += dt*(bb*M[0, j] + d*M[1, j])
                    M[0, j] += dt*M[2, j]; M[1, j] += dt*M[3, j]
            # explicit Lyapunov-equation update with the Jacobian at the new state
            x, y = s[0], s[1]; a, bb, d = jac(x, y)
            J = np.zeros((4, 4)); J[0, 2] = 1; J[1, 3] = 1; J[2, 0] = a; J[2, 1] = bb; J[3, 0] = bb; J[3, 1] = d
            C = C + dt*(J @ C + C @ J.T)
        sv = np.linalg.svd(L)[1]; ev = np.sort(np.linalg.eigvalsh(C))[::-1]
        ar_expl = np.sqrt(max(ev[1], 0.0)/ev[0]) if ev[0] > 0 else np.nan
        n0 = np.sqrt(np.sum(N[:, 0]**2)); n1 = np.sqrt(np.sum(N[:, 1]**2))
        N[:, 0] /= n0; N[:, 1] /= n1
        svn = np.linalg.svd(N)[1]
        sp = min(np.sqrt(np.sum((N[:, 0] + N[:, 1])**2)), np.sqrt(np.sum((N[:, 0] - N[:, 1])**2)))
        w0 = W[:, 0]/np.sqrt(np.sum(W[:, 0]**2)); w1 = W[:, 1]/np.sqrt(np.sum(W[:, 1]**2))
        sali = min(np.sqrt(np.sum((w0 + w1)**2)), np.sqrt(np.sum((w0 - w1)**2)))
        W[:, 0] = w0; W[:, 1] = w1
        ratio = np.sqrt(np.sum(L[:, 1]**2))/np.sqrt(np.sum(L[:, 0]**2))
        out[b, 0] = sv[1]/sv[0]; out[b, 1] = ar_expl; out[b, 2] = svn[1]/svn[0]; out[b, 3] = sp; out[b, 4] = sali; out[b, 5] = ratio; out[b, 6] = b
        # trace renormalization (uniform scaling: aspect ratios are unchanged)
        tr = np.trace(L @ L.T); L *= np.sqrt(2*sig*sig/tr)
        trc = np.trace(C)
        if trc > 0:
            C *= 2*sig*sig/trc
    return out


if __name__ == '__main__':
    res = {}
    for name, y0 in ORBITS.items():
        px = np.sqrt(2*E - y0*y0 + 2*y0**3/3)
        o = run(np.array([0.0, y0, px, 0.0]), NBLK, BLOCK, DT, SIG)
        res[name] = o
        t = (np.arange(NBLK) + 1)*BLOCK*DT; z = np.where(~(o[:, 1] > 0))[0]
        print(f'{name}: first explicit-update collapse at t = {t[z[0]] if z.size else None}; '
              f'||w2||/||w1|| at t=2000: {o[-1, 5]:.0f}; min SALI {np.nanmin(o[:, 4]):.1e}')
    out = os.path.join(os.path.dirname(__file__), '..', 'data', 'fig1_aspect_ratio.npz')
    np.savez(out, t=(np.arange(NBLK) + 1)*BLOCK*DT, cols=np.array(['AR_sqrt', 'AR_expl', 'AR_hat', 'SALI_pair', 'SALI', 'ratio', 'block']),
             **res)
