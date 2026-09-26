"""Data for Fig. 2: material probability contours on the Henon-Heiles section x = 0 (E = 1/8).

The p-contour (p = 0.6) of a Gaussian of width sigma = 2e-3 centred at the orbits R and C is followed under the
Poincare map (x = 0, px > 0), with adaptive insertion of nodes where image edges become long or sharply turning.
Each node is an orbit on the energy surface (Yoshida-4 leapfrog, dt = 0.01; section crossings by bisection).
Records, after n = 0, 4, 8, 12 returns, the contour relative to its centroid, L / L_hull, the number of nodes and
the mean return time.
Usage:  python compute_fig2_contours.py            -> ../data/fig2_contours.npz
"""
import numpy as np, numba as nb, os

E = 0.125; SIG = 2e-3; P = 0.6; N0 = 128; DT = 0.01; NMAX = 12; SNAP = (0, 4, 8, 12)
DMAX = 2.5e-4          # insert a node where an image edge is longer than this (section units) ...
AMAX = 0.25            # ... or where the contour turns by more than this angle (rad) at a node
NCAP = 20000
W1 = 1.0/(2.0 - 2.0**(1/3)); W0 = -2.0**(1/3)/(2.0 - 2.0**(1/3))
ORBITS = {'R': 0.1, 'C': -0.015}


@nb.njit(cache=True, inline='always')
def lf(s, h):
    x, y = s[0], s[1]
    s[2] -= 0.5*h*(x + 2*x*y); s[3] -= 0.5*h*(y + x*x - y*y)
    s[0] += h*s[2]; s[1] += h*s[3]
    x, y = s[0], s[1]
    s[2] -= 0.5*h*(x + 2*x*y); s[3] -= 0.5*h*(y + x*x - y*y)


@nb.njit(cache=True, inline='always')
def y4(s, dt):
    for c in (W1, W0, W1):
        lf(s, c*dt)


@nb.njit(cache=True)
def next_return(s, dt):
    """advance s to the next upward crossing of x = 0 (px > 0); returns elapsed time."""
    t = 0.0
    while True:
        prev = s.copy(); y4(s, dt); t += dt
        if prev[0] < 0.0 <= s[0] and s[2] > 0 and t > 10*dt:
            lo, hi = 0.0, dt
            for _ in range(50):                       # bisection on the step length from prev
                mid = 0.5*(lo + hi); q = prev.copy(); y4(q, mid)
                if q[0] < 0.0:
                    lo = mid
                else:
                    hi = mid
            q = prev.copy(); y4(q, 0.5*(lo + hi)); s[:] = q
            return t - dt + 0.5*(lo + hi)


@nb.njit(cache=True)
def orbit_returns(y, py, nret, dt):
    px = np.sqrt(max(2*E - py*py - y*y + 2*y**3/3, 0.0))
    s = np.array([0.0, y, px, py]); out = np.empty((nret + 1, 3)); out[0] = (y, py, 0.0); t = 0.0
    for n in range(1, nret + 1):
        t += next_return(s, dt); out[n, 0] = s[1]; out[n, 1] = s[3]; out[n, 2] = t
    return out


@nb.njit(parallel=True, cache=True)
def many(Y, PY, nret, dt):
    out = np.empty((Y.size, nret + 1, 3))
    for i in nb.prange(Y.size):
        out[i] = orbit_returns(Y[i], PY[i], nret, dt)
    return out


def hull_length(pts):
    """perimeter of the convex hull (monotone chain)."""
    p = sorted(map(tuple, pts))
    def half(seq):
        h = []
        for q in seq:
            while len(h) >= 2 and (h[-1][0]-h[-2][0])*(q[1]-h[-2][1]) - (h[-1][1]-h[-2][1])*(q[0]-h[-2][0]) <= 0:
                h.pop()
            h.append(q)
        return h
    hull = half(p)[:-1] + half(p[::-1])[:-1]; hull = np.array(hull + [hull[0]])
    return np.sum(np.linalg.norm(np.diff(hull, axis=0), axis=1))


def contour(y0):
    r = SIG*np.sqrt(-2*np.log(1 - P))
    th = np.linspace(0, 2*np.pi, N0, endpoint=False)
    traj = many(y0 + r*np.cos(th), r*np.sin(th), NMAX, DT)           # (N, NMAX+1, 3)
    snaps = {}
    for n in range(NMAX + 1):
        for _ in range(30):                                           # refine the contour at return n
            Z = traj[:, n, :2]; Zn = np.roll(Z, -1, axis=0); Zp = np.roll(Z, 1, axis=0)
            e = np.linalg.norm(Zn - Z, axis=1)
            u = Z - Zp; v = Zn - Z
            ang = np.abs(np.arctan2(u[:, 0]*v[:, 1] - u[:, 1]*v[:, 0], np.sum(u*v, axis=1)))
            bad = (e > DMAX) | (ang > AMAX) | (np.roll(ang, -1) > AMAX)
            if not bad.any() or len(th) > NCAP:
                break
            idx = np.where(bad)[0]; thn = np.roll(th, -1); thn[-1] += 2*np.pi
            new = 0.5*(th[idx] + thn[idx])
            nt = many(y0 + r*np.cos(new), r*np.sin(new), NMAX, DT)
            th = np.insert(th, idx + 1, new); traj = np.insert(traj, idx + 1, nt, axis=0)
        if n in SNAP:
            Z = traj[:, n, :2]; Zc = np.vstack([Z, Z[:1]])
            L = np.sum(np.linalg.norm(np.diff(Zc, axis=0), axis=1)); Lh = hull_length(Z)
            A = 0.5*abs(np.sum(Zc[:-1, 0]*Zc[1:, 1] - Zc[1:, 0]*Zc[:-1, 1]))
            snaps[n] = dict(pts=Z - Z.mean(axis=0), ratio=L/Lh, N=len(th), t=float(traj[:, n, 2].mean()), area=A)
            print(f'   n={n:2d} t={snaps[n]["t"]:5.1f} N={len(th):5d} L/Lh={L/Lh:.3f} A/A0={A/(np.pi*r*r):.4f}')
    return snaps


if __name__ == '__main__':
    out = {}
    for name, y0 in ORBITS.items():
        print(name)
        for n, v in contour(y0).items():
            out[f'{name}_{n}_pts'] = v['pts']
            out[f'{name}_{n}_meta'] = np.array([v['ratio'], v['N'], v['t'], v['area']])
    np.savez(os.path.join(os.path.dirname(__file__), '..', 'data', 'fig2_contours.npz'), snaps=np.array(SNAP), **out)
