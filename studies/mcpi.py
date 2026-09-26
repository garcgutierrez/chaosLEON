"""Vectorised Picard-Chebyshev (MCPI) propagation of a whole grid of Henon-Heiles states.
All grid points are advanced together: each Picard sweep is one force evaluation
at the N+1 Chebyshev-Gauss-Lobatto nodes for every grid point plus one matrix
product with the Chebyshev integration operator."""
import numpy as np, numpy.polynomial.chebyshev as C, time

def cheb_integration_matrix(N):
    tau = -np.cos(np.pi*np.arange(N+1)/N)                 # ascending CGL nodes on [-1,1]
    V = C.chebvander(tau, N)                               # V[i,k] = T_k(tau_i)
    Vint = np.empty_like(V)
    for k in range(N+1):
        e = np.zeros(N+1); e[k] = 1
        Vint[:, k] = C.chebval(tau, C.chebint(e, lbnd=-1))
    return tau, Vint @ np.linalg.inv(V)

def rhs(Z):
    x, y, px, py = Z[..., 0], Z[..., 1], Z[..., 2], Z[..., 3]
    return np.stack((px, py, -(x + 2*x*y), -(y + x*x - y*y)), -1)

def mcpi_segment(z0, dt_seg, I, guess=None, tol=1e-13, maxit=60):
    """z0: (M,4). Returns Z at all nodes (N+1,M,4), number of Picard sweeps."""
    n1 = I.shape[0]; M = z0.shape[0]
    Z = np.broadcast_to(z0, (n1, M, 4)).copy() if guess is None else guess
    for it in range(maxit):
        F = rhs(Z).reshape(n1, M*4)
        Zn = z0[None] + 0.5*dt_seg*(I @ F).reshape(n1, M, 4)
        err = np.max(np.abs(Zn - Z)); Z = Zn
        if err < tol:
            break
    return Z, it+1

def propagate(z0, T, dt_seg, N):
    tau, I = cheb_integration_matrix(N)
    z = z0.copy(); its = []
    for s in range(int(round(T/dt_seg))):
        Z, it = mcpi_segment(z, dt_seg, I); z = Z[-1].copy(); its.append(it)
    return z, np.mean(its)

if __name__ == '__main__':
    from newind import px_from
    from fullgrid import advance
    rng = np.random.default_rng(0); P = []
    while len(P) < 20000:
        y, py = rng.uniform(-.6, .8), rng.uniform(-.6, .6); px = px_from(y, py)
        if np.isfinite(px) and px > 1e-3: P.append((0., y, px, py))
    X = np.array(P); H0 = 0.5*(X[:, 2]**2+X[:, 3]**2+X[:, 0]**2+X[:, 1]**2)+X[:, 0]**2*X[:, 1]-X[:, 1]**3/3
    T = 20.0
    ref = X.copy(); advance(ref[:5], np.ones(5, np.bool_), 0.001, 1)
    ref = X.copy(); advance(ref, np.ones(len(X), np.bool_), 0.001, int(T/0.001))   # accurate reference
    Yo = X.copy(); t0 = time.time(); advance(Yo, np.ones(len(X), np.bool_), 0.01, int(T/0.01)); ty = time.time()-t0
    print(f'Yoshida dt=0.01 (numba, 2 threads): {ty:.2f}s, max err vs dt=1e-3: {np.max(np.abs(Yo-ref)):.1e}')
    for dts, N in [(1.0, 16), (2.0, 24), (2.5, 32), (5.0, 48)]:
        t0 = time.time(); z, it = propagate(X, T, dts, N); tm = time.time()-t0
        Hf = 0.5*(z[:, 2]**2+z[:, 3]**2+z[:, 0]**2+z[:, 1]**2)+z[:, 0]**2*z[:, 1]-z[:, 1]**3/3
        print(f'MCPI seg={dts} N={N}: {tm:.2f}s, mean sweeps {it:.1f}, max err {np.max(np.abs(z-ref)):.1e}, max |dH| {np.max(np.abs(Hf-H0)):.1e}')
