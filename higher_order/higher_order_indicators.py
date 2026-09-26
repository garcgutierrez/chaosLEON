"""Third- and fourth-order extension of the uncertainty-kernel chaos indicators.

Usage:
  python higher_order_indicators.py verify                       # jet/variational checks, all systems
  python higher_order_indicators.py run hh x0 0.125 1000 2000 ho_hh_0125.npz     (system, IC plane, E, grid N, nodes, out)
  python higher_order_indicators.py run 3dof q0 0.07 1000 1500 ho_3dof_007.npz
  python higher_order_indicators.py run gal rz -0.75 1000 2000 ho_gal.npz
  systems/planes: see PLANES (hh: x0, y0; 3dof: q0, p0; gal: rz, rpr; qo: x0)
  python higher_order_indicators.py analyze ho_*.npz

A material segment through x0 along a grid coordinate xi is mapped to
    x(t, xi) = phi_t(x0) + sum_k xi^k Z_k(t)/k!,   k = 1..4,
where Z_k are the k-th variations (exact discrete variations of the Yoshida-4 leapfrog,
Faa di Bruno up to 4th order) and the initial jets Z_k(0) are those of the
iso-energetic initial-condition curve (so that the grid is the h -> 0 limit).

Grid side: a 9-point cross stencil (+-h, +-2h per direction) gives the EXACT quartic
collocation surrogate of each kernel,  x(t, xi) ~ sum_k c_k(t) xi^k  (xi in units of h),
    c1 = [8(x1-x-1) - (x2-x-2)]/12,        c2 = [-x2 + 16x1 - 30x0 + 16x-1 - x-2]/24,
    c3 = [x2 - 2x1 + 2x-1 - x-2]/12,        c4 = [x2 - 4x1 + 6x0 - 4x-1 + x-2]/24,
with c_k -> h^k Z_k/k!.  Still one trajectory per grid node.

Indicators (sup over read-out times t <= T, section projection P at the centre):
  OFLI_K   : log|P sum_{k<=K} Z_k/k!|                      (K=2 is Barrio's OFLI2)
  Lam_k    : (1/(k-1)) log(|PZ_1 ^ PZ_k| / (k! |PZ_1|^2))   = -log of the scale at which the
             k-th order bending equals the linear term (k=2: B - log 2 of the paper);
             Lam_max = max_k Lam_k is a Cauchy-Hadamard (radius of validity) estimate.
  eta_K    : log-odds of the non-Gaussian variance fraction of the pushforward of a Gaussian
             kernel of std sigma (=h) through the degree-K surrogate, in probabilists' Hermite
             modes:  sum_{k>=2} k!|a_k|^2 / |a_1|^2.
"""
import numpy as np, numba as nb, math, sys, time, json

Y1 = 1.0/(2.0-2.0**(1.0/3.0)); Y0 = -2.0**(1.0/3.0)/(2.0-2.0**(1.0/3.0))
AG = 0.05831
OM3 = np.array([1.0, 1.4142, 1.7321])
LOGF = np.log(np.array([1., 1., 2., 6., 24.]))
KIND = {'hh': 0, '3dof': 1, 'gal': 2, 'qo': 3}     # qo: coupled quartic oscillators, V = (x^2+y^2)/2 + x^2 y^2
DIM = (4, 6, 4, 4)

TCOLS = ['OFLI', 'OFLI2', 'OFLI3', 'OFLI4', 'Lam2', 'Lam3', 'Lam4', 'Lammax', 'eta2', 'eta3', 'eta4']
GCOLS = ['S_h', 'OFLI2_h(3pt)', 'B_h(3pt)', 'OFLI2_h', 'OFLI3_h', 'OFLI4_h', 'Lam2_h', 'Lam3_h', 'Lam4_h',
         'Lammax_h', 'eta2_h', 'eta3_h', 'eta4_h']
NT, NG = len(TCOLS), len(GCOLS)


# ----------------------------------------------------------------------------- dynamics
@nb.njit(cache=True, inline='always')
def kick_orbit(kind, s, hk, om):
    if kind == 0:
        x, y = s[0], s[1]
        s[2] -= hk*(x + 2*x*y); s[3] -= hk*(y + x*x - y*y)
    elif kind == 1:
        q0, q1, q2 = s[0], s[1], s[2]
        s[3] -= hk*(om[0]*q0 + 2*q0*q1 + 2*q0*q2); s[4] -= hk*(om[1]*q1 + q0*q0); s[5] -= hk*(om[2]*q2 + q0*q0)
    elif kind == 2:
        r, z = s[0], s[1]; r2 = r*r + z*z
        s[2] -= hk*(r/r2 - AG/(r*r*r)); s[3] -= hk*(z/r2 - 1.0)
    else:
        x, y = s[0], s[1]
        s[2] -= hk*(x + 2*x*y*y); s[3] -= hk*(y + 2*x*x*y)


@nb.njit(cache=True, inline='always')
def drift(s, h, om, d):
    for i in range(d):
        s[i] += h*om[i]*s[d+i]


@nb.njit(cache=True)
def step_orbits(kind, S, dt, n, om):
    d = S.shape[1]//2
    for _ in range(n):
        for c in (Y1, Y0, Y1):
            h = c*dt
            for k in range(S.shape[0]):
                kick_orbit(kind, S[k], 0.5*h, om); drift(S[k], h, om, d); kick_orbit(kind, S[k], 0.5*h, om)


@nb.njit(cache=True, inline='always')
def gal_hess(r, z):
    r2 = r*r + z*z; r4 = r2*r2
    return (z*z - r*r)/r4 + 3*AG/r**4, -2*r*z/r4, (r*r - z*z)/r4


@nb.njit(cache=True, inline='always')
def kick_tan(kind, s, W, hk, om):
    """W[j] are deviation vectors; momentum parts get hk * DF(q) W[j]_q."""
    if kind == 0:
        x, y = s[0], s[1]; A = 1 + 2*y; B = 2*x; D = 1 - 2*y
        for j in range(W.shape[0]):
            a0, a1 = W[j, 0], W[j, 1]
            W[j, 2] -= hk*(A*a0 + B*a1); W[j, 3] -= hk*(B*a0 + D*a1)
    elif kind == 1:
        A = om[0] + 2*s[1] + 2*s[2]; B = 2*s[0]
        for j in range(W.shape[0]):
            a0, a1, a2 = W[j, 0], W[j, 1], W[j, 2]
            W[j, 3] -= hk*(A*a0 + B*a1 + B*a2); W[j, 4] -= hk*(B*a0 + om[1]*a1); W[j, 5] -= hk*(B*a0 + om[2]*a2)
    else:
        if kind == 2:
            A, B, D = gal_hess(s[0], s[1])
        else:
            x, y = s[0], s[1]; A = 1 + 2*y*y; B = 4*x*y; D = 1 + 2*x*x
        for j in range(W.shape[0]):
            a0, a1 = W[j, 0], W[j, 1]
            W[j, 2] -= hk*(A*a0 + B*a1); W[j, 3] -= hk*(B*a0 + D*a1)


@nb.njit(cache=True)
def step_tan(kind, s, W, dt, n, om):
    d = s.size//2
    for _ in range(n):
        for c in (Y1, Y0, Y1):
            h = c*dt
            for half in range(2):
                kick_tan(kind, s, W, 0.5*h, om); kick_orbit(kind, s, 0.5*h, om)
                if half == 0:
                    drift(s, h, om, d)
                    for j in range(W.shape[0]):
                        drift(W[j], h, om, d)


# --- jets (orders 1..4) ------------------------------------------------------------------
@nb.njit(cache=True)
def gal_tab(r, z, T):
    """T[n, nz] = d^n V / dr^(n-nz) dz^nz for n = 2..5."""
    w = complex(r, z); iw = 1.0/w; fact = 1.0
    for n in range(2, 6):
        fact *= (n - 1)
        base = (-1.0)**(n - 1)*fact*iw**n; ip = 1.0 + 0j
        for nz in range(n + 1):
            T[n, nz] = (ip*base).real; ip *= 1j
        T[n, 0] += 0.5*AG*(-1.0)**n*math.gamma(n + 2)/r**(n + 2)


@nb.njit(cache=True, inline='always')
def gal_ml(T, k, i, a, b, c, e):
    """D^k F_i [a, b, c, e] (first k of the vectors used), F = -grad V."""
    acc = 0.0
    if k == 1:
        for j in range(2):
            acc += T[2, i+j]*a[j]
    elif k == 2:
        for j in range(2):
            for l in range(2):
                acc += T[3, i+j+l]*a[j]*b[l]
    elif k == 3:
        for j in range(2):
            for l in range(2):
                for m in range(2):
                    acc += T[4, i+j+l+m]*a[j]*b[l]*c[m]
    else:
        for j in range(2):
            for l in range(2):
                for m in range(2):
                    for o in range(2):
                        acc += T[5, i+j+l+m+o]*a[j]*b[l]*c[m]*e[o]
    return -acc


@nb.njit(cache=True, inline='always')
def hh_df(x, y, a):
    return (-(1 + 2*y)*a[0] - 2*x*a[1], -2*x*a[0] - (1 - 2*y)*a[1])


@nb.njit(cache=True, inline='always')
def hh_d2(a, b):
    return (-2*(a[0]*b[1] + a[1]*b[0]), -2*a[0]*b[0] + 2*a[1]*b[1])


@nb.njit(cache=True, inline='always')
def d3_df(q, a, om):
    A = om[0] + 2*q[1] + 2*q[2]; B = 2*q[0]
    return (-(A*a[0] + B*a[1] + B*a[2]), -(B*a[0] + om[1]*a[1]), -(B*a[0] + om[2]*a[2]))


@nb.njit(cache=True, inline='always')
def d3_d2(a, b):
    t = -2*a[0]*b[0]
    return (-2*(a[0]*b[1] + a[1]*b[0] + a[0]*b[2] + a[2]*b[0]), t, t)


@nb.njit(cache=True, inline='always')
def qo_df(x, y, a):
    return (-(1 + 2*y*y)*a[0] - 4*x*y*a[1], -4*x*y*a[0] - (1 + 2*x*x)*a[1])


@nb.njit(cache=True, inline='always')
def qo_d2(x, y, a, b):
    return (-(4*y*(a[0]*b[1] + a[1]*b[0]) + 4*x*a[1]*b[1]), -(4*y*a[0]*b[0] + 4*x*(a[0]*b[1] + a[1]*b[0])))


@nb.njit(cache=True, inline='always')
def qo_d3(a, b, c):
    return (-4*(a[0]*b[1]*c[1] + a[1]*b[0]*c[1] + a[1]*b[1]*c[0]), -4*(a[1]*b[0]*c[0] + a[0]*b[1]*c[0] + a[0]*b[0]*c[1]))


@nb.njit(cache=True)
def kick_jet(kind, s, Z, hk, om, T):
    """Z[0..3] = first..fourth variations. Faa di Bruno for F(q(xi)):
       g1 = DF w, g2 = DF v + D2F[w,w], g3 = DF z3 + 3D2F[w,v] + D3F[w,w,w],
       g4 = DF z4 + 4D2F[w,z3] + 3D2F[v,v] + 6D3F[w,w,v] + D4F[w,w,w,w]."""
    w = Z[0]; v = Z[1]; z3 = Z[2]; z4 = Z[3]
    if kind == 0:
        x, y = s[0], s[1]
        a = hh_df(x, y, w)
        b = hh_df(x, y, v); b2 = hh_d2(w, w)
        c = hh_df(x, y, z3); c2 = hh_d2(w, v)
        e = hh_df(x, y, z4); e2 = hh_d2(w, z3); e3 = hh_d2(v, v)
        for i in range(2):
            Z[0, 2+i] += hk*a[i]; Z[1, 2+i] += hk*(b[i] + b2[i]); Z[2, 2+i] += hk*(c[i] + 3*c2[i])
            Z[3, 2+i] += hk*(e[i] + 4*e2[i] + 3*e3[i])
    elif kind == 1:
        a = d3_df(s, w, om)
        b = d3_df(s, v, om); b2 = d3_d2(w, w)
        c = d3_df(s, z3, om); c2 = d3_d2(w, v)
        e = d3_df(s, z4, om); e2 = d3_d2(w, z3); e3 = d3_d2(v, v)
        for i in range(3):
            Z[0, 3+i] += hk*a[i]; Z[1, 3+i] += hk*(b[i] + b2[i]); Z[2, 3+i] += hk*(c[i] + 3*c2[i])
            Z[3, 3+i] += hk*(e[i] + 4*e2[i] + 3*e3[i])
    elif kind == 3:
        x, y = s[0], s[1]
        a = qo_df(x, y, w)
        b = qo_df(x, y, v); b2 = qo_d2(x, y, w, w)
        c = qo_df(x, y, z3); c2 = qo_d2(x, y, w, v); c3 = qo_d3(w, w, w)
        e = qo_df(x, y, z4); e2 = qo_d2(x, y, w, z3); e3 = qo_d2(x, y, v, v); e4 = qo_d3(w, w, v)
        for i in range(2):
            Z[0, 2+i] += hk*a[i]; Z[1, 2+i] += hk*(b[i] + b2[i]); Z[2, 2+i] += hk*(c[i] + 3*c2[i] + c3[i])
            Z[3, 2+i] += hk*(e[i] + 4*e2[i] + 3*e3[i] + 6*e4[i])
    else:
        gal_tab(s[0], s[1], T)
        g = np.empty((4, 2))
        for i in range(2):
            g[0, i] = gal_ml(T, 1, i, w, w, w, w)
            g[1, i] = gal_ml(T, 1, i, v, v, v, v) + gal_ml(T, 2, i, w, w, w, w)
            g[2, i] = gal_ml(T, 1, i, z3, z3, z3, z3) + 3*gal_ml(T, 2, i, w, v, v, v) + gal_ml(T, 3, i, w, w, w, w)
            g[3, i] = (gal_ml(T, 1, i, z4, z4, z4, z4) + 4*gal_ml(T, 2, i, w, z3, z3, z3) + 3*gal_ml(T, 2, i, v, v, v, v)
                       + 6*gal_ml(T, 3, i, w, w, v, v) + gal_ml(T, 4, i, w, w, w, w))
        for k in range(4):
            for i in range(2):
                Z[k, 2+i] += hk*g[k, i]


@nb.njit(cache=True)
def step_jet(kind, s, Z, dt, n, om, T):
    d = s.size//2
    for _ in range(n):
        for c in (Y1, Y0, Y1):
            h = c*dt
            for half in range(2):
                kick_jet(kind, s, Z, 0.5*h, om, T); kick_orbit(kind, s, 0.5*h, om)
                if half == 0:
                    drift(s, h, om, d)
                    for k in range(4):
                        drift(Z[k], h, om, d)


# ----------------------------------------------------------------------------- geometry
@nb.njit(cache=True)
def proj(kind, s, x, om):
    """remove components along grad H and the flow f (P of the paper)."""
    d = s.size//2; n = s.size; F = np.zeros(d); t = s.copy()
    for i in range(d):
        t[d+i] = 0.0
    kick_orbit(kind, t, -1.0, om)          # t[d:] = grad V
    g = np.empty(n); f = np.empty(n)
    for i in range(d):
        g[i] = t[d+i]; g[d+i] = om[i]*s[d+i]; f[i] = om[i]*s[d+i]; f[d+i] = -t[d+i]
    g /= math.sqrt(np.sum(g*g)); f -= np.dot(f, g)*g; f /= math.sqrt(np.sum(f*f))
    return x - np.dot(x, g)*g - np.dot(x, f)*f


@nb.njit(cache=True, inline='always')
def nrm(a):
    return math.sqrt(np.sum(a*a))


@nb.njit(cache=True)
def wedge(a, b):
    na2 = np.sum(a*a)
    if na2 == 0.0:
        return 0.0
    bp = b - (np.dot(a, b)/na2)*a
    return math.sqrt(na2)*nrm(bp)


@nb.njit(cache=True)
def sali(a, b):
    a = a/nrm(a); b = b/nrm(b)
    return min(nrm(a + b), nrm(a - b))


@nb.njit(cache=True)
def eta_scores(m1, m2, m3, m4, out):
    """log-odds non-Gaussian/linear variance for degree 2, 3, 4 surrogates sum m_j xi^j."""
    lin = np.sum(m1*m1); nl = 2*np.sum(m2*m2)
    out[0] = math.log(max(nl, 1e-300)) - math.log(max(lin, 1e-300))
    a1 = m1 + 3*m3; lin = np.sum(a1*a1); nl = 2*np.sum(m2*m2) + 6*np.sum(m3*m3)
    out[1] = math.log(max(nl, 1e-300)) - math.log(max(lin, 1e-300))
    a2 = m2 + 6*m4; nl = 2*np.sum(a2*a2) + 6*np.sum(m3*m3) + 24*np.sum(m4*m4)
    out[2] = math.log(max(nl, 1e-300)) - math.log(max(lin, 1e-300))


# ----------------------------------------------------------------------------- engines
@nb.njit(cache=True)
def tangent_engine(kind, x0, Z0, om, dt, tau, nbk, hidx, sig):
    """Jets along one direction; cols TCOLS."""
    s = x0.copy(); Z = Z0.copy(); T = np.zeros((6, 6)); Lc = 0.0
    sup = np.full(NT, -1e300); out = np.full((hidx.size, NT), np.nan); hk = 0; e3 = np.empty(3)
    for b in range(nbk):
        step_jet(kind, s, Z, dt, tau, om, T)
        p0 = proj(kind, s, Z[0], om); p1 = proj(kind, s, Z[1], om); p2 = proj(kind, s, Z[2], om); p3 = proj(kind, s, Z[3], om)
        P = (p0, p1, p2, p3); n1 = nrm(p0)
        sup[0] = max(sup[0], Lc + math.log(n1))
        for K in range(2, 5):                       # OFLI_K, stable evaluation of sum e^{(k-1)Lc} p_k/k!
            M = -1e300
            for k in range(K):
                M = max(M, k*Lc + math.log(max(nrm(P[k]), 1e-300)) - LOGF[k+1])
            acc = np.zeros(s.size)
            for k in range(K):
                acc += math.exp(k*Lc - M - LOGF[k+1])*P[k]
            sup[K-1] = max(sup[K-1], Lc + M + math.log(max(nrm(acc), 1e-300)))
        lm = -1e300
        for k in range(2, 5):                       # Lam_k
            v = Lc + (math.log(max(wedge(p0, P[k-1]), 1e-300)) - 2*math.log(n1) - LOGF[k])/(k - 1)
            sup[2+k] = max(sup[2+k], v); lm = max(lm, v)
        sup[7] = max(sup[7], lm)
        ell = math.log(sig) + Lc; top = 4 if ell > 0 else 1   # m_j = r^j p_j/j!, rescaled by r^-top
        eta_scores(math.exp((1-top)*ell)*p0, math.exp((2-top)*ell - LOGF[2])*p1,
                   math.exp((3-top)*ell - LOGF[3])*p2, math.exp((4-top)*ell - LOGF[4])*p3, e3)
        for j in range(3):
            sup[8+j] = max(sup[8+j], e3[j])
        if hk < hidx.size and b == hidx[hk]:
            out[hk] = sup; hk += 1
        c = nrm(Z[0]); Lc += math.log(c)
        for k in range(4):
            Z[k] /= c**(k+1)
    return out


@nb.njit(cache=True)
def sali_engine(kind, x0, W0, om, dt, tau, nbk, hidx, nbref):
    """SALI at the horizons and the reference SALI(t = nbref*tau*dt)."""
    s = x0.copy(); W = W0.copy(); out = np.full(hidx.size, np.nan); hk = 0; val = 1.0
    for b in range(nbref):
        step_tan(kind, s, W, dt, tau, om); val = sali(W[0], W[1])
        if hk < hidx.size and b == hidx[hk]:
            out[hk] = val; hk += 1
        W[0] /= nrm(W[0]); W[1] /= nrm(W[1])
        if b >= nbk and val < 1e-12:
            break
    return out, val


@nb.njit(cache=True)
def quartic(x0, xp1, xm1, xp2, xm2):
    c1 = (8*(xp1 - xm1) - (xp2 - xm2))/12
    c2 = (-xp2 + 16*xp1 - 30*x0 + 16*xm1 - xm2)/24
    c3 = (xp2 - 2*xp1 + 2*xm1 - xm2)/12
    c4 = (xp2 - 4*xp1 + 6*x0 - 4*xm1 + xm2)/24
    return c1, c2, c3, c4


@nb.njit(cache=True)
def grid_engine(kind, S0, h1, h2, om, dt, tau, nbk, hidx):
    """S0 (9, n): centre, +h1, -h1, +2h1, -2h1, +h2, -h2, +2h2, -2h2. cols GCOLS."""
    S = S0.copy(); sup = np.full(NG, -1e300); out = np.full((hidx.size, NG), np.nan); hk = 0; e3 = np.empty(3)
    for b in range(nbk):
        step_orbits(kind, S, dt, tau, om)
        x0 = S[0]; cur = np.full(NG, -1e300)
        for dd in range(2):
            o = 1 + 4*dd; h = h1 if dd == 0 else h2
            xp1, xm1, xp2, xm2 = S[o], S[o+1], S[o+2], S[o+3]
            # paper (3-point) versions
            D = (xp1 - xm1)/(2*h); C = (xp1 + xm1 - 2*x0)/(h*h)
            pD = proj(kind, x0, D, om); pC = proj(kind, x0, C, om); nD = nrm(pD)
            cur[0] = max(cur[0], math.log(nD))
            cur[1] = max(cur[1], math.log(nrm(proj(kind, x0, D + 0.5*C, om))))
            cur[2] = max(cur[2], math.log(max(h*wedge(pD, pC)/(nD*nD), 1e-300)))
            # quartic collocation surrogate
            c1, c2, c3, c4 = quartic(x0, xp1, xm1, xp2, xm2)
            q1 = proj(kind, x0, c1, om); q2 = proj(kind, x0, c2, om); q3 = proj(kind, x0, c3, om); q4 = proj(kind, x0, c4, om)
            u1 = q1/h; u2 = q2/h**2; u3 = q3/h**3; u4 = q4/h**4
            cur[3] = max(cur[3], math.log(nrm(u1 + u2)))
            cur[4] = max(cur[4], math.log(nrm(u1 + u2 + u3)))
            cur[5] = max(cur[5], math.log(nrm(u1 + u2 + u3 + u4)))
            n1 = nrm(q1); Q = (q1, q2, q3, q4); lm = -1e300
            for k in range(2, 5):
                v = math.log(max(wedge(q1, Q[k-1]), 1e-300)/(n1*n1))/(k - 1)
                cur[4+k] = max(cur[4+k], v); lm = max(lm, v)
            cur[9] = max(cur[9], lm)
            eta_scores(q1, q2, q3, q4, e3)
            for j in range(3):
                cur[10+j] = max(cur[10+j], e3[j])
        for j in range(NG):
            sup[j] = max(sup[j], cur[j])
        if hk < hidx.size and b == hidx[hk]:
            out[hk] = sup; hk += 1
    return out


@nb.njit(parallel=True, cache=True)
def run_all(kind, S0all, Z0all, W0, h1, h2, om, dt, tau, nbk, hidx, nbref):
    n = S0all.shape[0]
    G = np.empty((n, hidx.size, NG)); Tn = np.empty((n, 2, hidx.size, NT)); SA = np.empty((n, hidx.size)); ref = np.empty(n)
    for i in nb.prange(n):
        G[i] = grid_engine(kind, S0all[i], h1, h2, om, dt, tau, nbk, hidx)
        for dd in range(2):
            Tn[i, dd] = tangent_engine(kind, S0all[i, 0], Z0all[i, dd], om, dt, tau, nbk, hidx, h1 if dd == 0 else h2)
        SA[i], ref[i] = sali_engine(kind, S0all[i, 0], W0, om, dt, tau, nbk, hidx, nbref)
    return G, Tn, SA, ref


# ----------------------------------------------------------------------------- initial conditions
def sqrt_jet(g):
    """Taylor coefficients of sqrt(g(xi)) up to xi^4."""
    u = np.zeros(5); u[0] = math.sqrt(g[0])
    for k in range(1, 5):
        u[k] = (g[k] - sum(u[j]*u[k-j] for j in range(1, k)))/(2*u[0])
    return u


def ic_state(sysname, E, a, b):
    """grid coordinates (a, b) -> admissible state or None."""
    if sysname == 'hh':
        v = 2*E - b*b - a*a + 2*a**3/3
        return None if v < 1e-6 else np.array([0., a, math.sqrt(v), b])
    if sysname == '3dof':
        v = 2*E - OM3[0]*a*a - OM3[1]*b*b
        return None if v <= 1e-6 else np.array([0., 0, 0, a, b, math.sqrt(v/OM3[2])])
    if a <= 0:
        return None
    v = 2*(E - 0.5*math.log(a*a + b*b) - AG/(2*a*a) + b)
    return None if v < 1e-6 else np.array([a, b, math.sqrt(v), 0.])


def ic_jet(sysname, E, a, b, dd):
    """Z_k(0), k=1..4: xi-derivatives of the iso-energetic IC curve along grid direction dd."""
    n = 6 if sysname == '3dof' else 4; Z = np.zeros((4, n)); g = np.zeros(5)
    if sysname == 'hh':
        g[0] = 2*E - b*b - a*a + 2*a**3/3
        if dd == 0:
            g[1] = -2*a + 2*a*a; g[2] = -1 + 2*a; g[3] = 2/3; Z[0, 1] = 1
        else:
            g[1] = -2*b; g[2] = -1; Z[0, 3] = 1
        u = sqrt_jet(g); slot = 2
    elif sysname == '3dof':
        o = OM3; g[0] = (2*E - o[0]*a*a - o[1]*b*b)/o[2]
        if dd == 0:
            g[1] = -2*o[0]*a/o[2]; g[2] = -o[0]/o[2]; Z[0, 3] = 1
        else:
            g[1] = -2*o[1]*b/o[2]; g[2] = -o[1]/o[2]; Z[0, 4] = 1
        u = sqrt_jet(g); slot = 5
    else:
        w = complex(a, b); g[0] = 2*(E - 0.5*math.log(a*a + b*b) - AG/(2*a*a) + b)
        for k in range(1, 5):
            step = 1.0 if dd == 0 else 1j
            Lk = 2*((-1)**(k-1)*step**k/(k*w**k)).real          # ln(r^2+z^2) coefficients
            g[k] = -Lk
            if dd == 0:
                g[k] -= AG*(k+1)*(-1)**k/a**(k+2)
        if dd == 1:
            g[1] += 2.0
        Z[0, dd] = 1; u = sqrt_jet(g); slot = 2
    for k in range(1, 5):
        Z[k-1, slot] = math.factorial(k)*u[k]
    return Z


def _sq(v):
    return np.sqrt(v + 0j)


def _vgal(r, z):
    return 0.5*np.log(r*r + z*z) + AG/(2*r*r) - z


# (system, plane) -> (grid ranges as a function of E, IC map (a, b, E) -> state [complex-safe], index of the
# momentum fixed by the energy).  'x0'/'q0'/'rz' are the planes of the paper; the others were added to test
# the dependence on the initial-condition plane.
PLANES = {
    ('hh', 'x0'): (lambda E: ((-0.6, 0.8), (-0.6, 0.6)),
                   lambda a, b, E: [0*a, a, _sq(2*E - b*b - a*a + 2*a**3/3), b], 2),
    ('hh', 'y0'): (lambda E: ((-0.6, 0.6), (-0.6, 0.6)),
                   lambda a, b, E: [a, 0*a, b, _sq(2*E - b*b - a*a)], 3),
    ('3dof', 'q0'): (lambda E: ((-np.sqrt(2*E/OM3[0]), np.sqrt(2*E/OM3[0])), (-np.sqrt(2*E/OM3[1]), np.sqrt(2*E/OM3[1]))),
                     lambda a, b, E: [0*a, 0*a, 0*a, a, b, _sq((2*E - OM3[0]*a*a - OM3[1]*b*b)/OM3[2])], 5),
    ('3dof', 'p0'): (lambda E: ((-1.1*np.sqrt(2*E/OM3[0]), 1.1*np.sqrt(2*E/OM3[0])), (-1.2*np.sqrt(2*E/OM3[1]), 1.2*np.sqrt(2*E/OM3[1]))),
                     lambda a, b, E: [a, b, 0*a, 0*a, 0*a, _sq((2*E - OM3[0]*a*a - OM3[1]*b*b - 2*a*a*b)/OM3[2])], 5),
    ('gal', 'rz'): (lambda E: ((0.1, 0.5), (-0.1, 0.6)),
                    lambda a, b, E: [a, b, _sq(2*(E - _vgal(a, b))), 0*a], 2),
    ('gal', 'rpr'): (lambda E: ((0.15, 0.46), (-0.65, 0.65)),
                     lambda a, b, E: [a, 0*a + 0.1, b, _sq(2*(E - _vgal(a, 0.1)) - b*b)], 3),
    ('qo', 'x0'): (lambda E: ((-np.sqrt(2*E), np.sqrt(2*E)), (-np.sqrt(2*E), np.sqrt(2*E))),
                   lambda a, b, E: [0*a, a, _sq(2*E - b*b - a*a), b], 2),
}


def plane_state(sysname, plane, E, a, b):
    """real admissible state or None (energy-fixed momentum >= 1e-3, as in the paper's scripts)."""
    _, f, pi = PLANES[(sysname, plane)]
    x = np.array(f(complex(a), complex(b), E))
    if not np.all(np.isfinite(x)) or abs(x[pi].imag) > 0 or x[pi].real < 1e-3:
        return None
    return x.real.copy()


def plane_jet(sysname, plane, E, a, b, dd, N=64):
    r"""Z_k(0), k=1..4, of the iso-energetic IC curve along grid direction dd, from the Cauchy integral
    f_k = (1/2 pi i) \oint f(xi) xi^{-k-1} dxi on a circle whose radius stays well inside the nearest singularity
    (root of the energy-fixed momentum, or rho = 0)."""
    _, f, pi = PLANES[(sysname, plane)]
    x0 = np.array(f(complex(a), complex(b), E)).real; eps = 1e-6
    xp = np.array(f(complex(a + eps*(dd == 0)), complex(b + eps*(dd == 1)), E)).real
    dp = abs(xp[pi] - x0[pi])/eps
    rad = min(0.02, 0.25*x0[pi]/(2*dp + 1e-12))
    if sysname == 'gal':
        rad = min(rad, 0.25*a)
    th = 2*np.pi*np.arange(N)/N; xi = rad*np.exp(1j*th)
    F = np.array(f(a + xi*(dd == 0), b + xi*(dd == 1), E))                     # (n, N)
    Z = np.empty((4, F.shape[0]))
    for k in range(1, 5):
        Z[k-1] = math.factorial(k)*(F*np.exp(-1j*k*th)).mean(axis=1).real/rad**k
    return Z


def sample(sysname, plane, E, Ngrid, nsamp, seed=5):
    ra, rb = PLANES[(sysname, plane)][0](E)
    av = np.linspace(*ra, Ngrid); bv = np.linspace(*rb, Ngrid); h1, h2 = av[1]-av[0], bv[1]-bv[0]
    rng = np.random.default_rng(seed); S0, Z0 = [], []
    offs = ((0, 0), (1, 0), (-1, 0), (2, 0), (-2, 0), (0, 1), (0, -1), (0, 2), (0, -2))
    tries = 0
    while len(S0) < nsamp:
        tries += 1
        if tries > 200*nsamp:
            raise RuntimeError('too few admissible nodes')
        i, j = rng.integers(2, Ngrid-2, 2); st = []
        for oa, ob in offs:
            x = plane_state(sysname, plane, E, av[i] + oa*h1, bv[j] + ob*h2)
            if x is None:
                break
            st.append(x)
        if len(st) == 9:
            S0.append(st); Z0.append([plane_jet(sysname, plane, E, av[i], bv[j], 0), plane_jet(sysname, plane, E, av[i], bv[j], 1)])
    return np.array(S0), np.array(Z0), h1, h2


def setup(sysname):
    """om, dt, tau, horizons, SALI initial vectors (as in the paper's scripts)."""
    if sysname in ('hh', 'qo'):
        W0 = np.zeros((2, 4)); W0[0, 0] = 1; W0[1, 2] = 1
        return np.ones(2), 0.01, 500, np.array([25, 50, 100, 150, 200, 300, 500]), W0
    if sysname == '3dof':
        W0 = np.random.default_rng(1).standard_normal((6, 3))[:, :2].T.copy()
        W0 /= np.linalg.norm(W0, axis=1)[:, None]
        return OM3.copy(), 0.01, 500, np.array([100, 200, 500, 1000, 2000]), W0
    W0 = np.zeros((2, 4)); W0[0, 0] = 1; W0[1, 1] = 1
    return np.ones(2), 0.005, 1000, np.array([25, 50, 100, 200, 300, 500, 1000]), W0


# ----------------------------------------------------------------------------- checks
VERIFY_POINTS = (('hh', 'x0', 0.125, (-0.0135, 0.0)), ('hh', 'y0', 0.125, (0.1, 0.2)),
                 ('3dof', 'q0', 0.07, (0.05, 0.1)), ('3dof', 'p0', 0.07, (0.1, 0.05)),
                 ('gal', 'rz', -0.75, (0.25, 0.2)), ('gal', 'rpr', -0.75, (0.3, 0.1)), ('qo', 'x0', 1.0, (0.3, 0.2)))


def verify():
    """(i) Cauchy IC jets vs the analytic ones (paper planes) and vs 5-point FD of the IC map;
    (ii) propagated jets vs Taylor remainder O(xi^5)."""
    for sysname, plane, E, (a, b) in VERIFY_POINTS:
        kind = KIND[sysname]; om, dt, tau, _, _ = setup(sysname)
        for dd in range(2):
            Z0 = plane_jet(sysname, plane, E, a, b, dd); d = 1e-3
            ana = ''
            if plane in ('x0', 'q0', 'rz') and sysname != 'qo':
                Za = ic_jet(sysname, E, a, b, dd)
                ana = 'vs analytic ' + ' '.join(f'{np.linalg.norm(Z0[k]-Za[k])/max(np.linalg.norm(Za[k]), 1e-12):.0e}' for k in range(4)) + ' | '
            X = [plane_state(sysname, plane, E, a + (k*d if dd == 0 else 0), b + (k*d if dd == 1 else 0)) for k in (0, 1, -1, 2, -2)]
            c = quartic(*X); fd = np.array([ci/d**(k+1)*math.factorial(k+1) for k, ci in enumerate(c)])
            e_ic = [np.linalg.norm(fd[k] - Z0[k])/max(np.linalg.norm(Z0[k]), 1e-12) for k in range(3)]
            s = X[0].copy(); Z = Z0.copy(); step_jet(kind, s, Z, dt, int(round(20/dt)), om, np.zeros((6, 6)))
            errs = []
            for delta in (4e-3, 2e-3, 1e-3):
                x1 = plane_state(sysname, plane, E, a + (delta if dd == 0 else 0), b + (delta if dd == 1 else 0))
                S = x1[None].copy(); step_orbits(kind, S, dt, int(round(20/dt)), om)
                tay = s + sum(delta**(k+1)*Z[k]/math.factorial(k+1) for k in range(4))
                tay3 = s + sum(delta**(k+1)*Z[k]/math.factorial(k+1) for k in range(3))
                errs.append((np.linalg.norm(S[0] - tay3), np.linalg.norm(S[0] - tay)))
            r3 = errs[0][0]/errs[1][0]; r4 = errs[1][1]/errs[2][1]
            print(f'{sysname:4s} {plane:3s} dir{dd}: {ana}FD k=1..3 {e_ic[0]:.0e} {e_ic[1]:.0e} {e_ic[2]:.0e} | '
                  f'remainder ratio cubic {r3:.1f} (16), quartic {r4:.1f} (32)')


# ----------------------------------------------------------------------------- analysis
def auc(y, z):
    z = np.asarray(z, float); o = np.argsort(z, kind='mergesort'); r = np.empty(len(z)); zs = z[o]; i = 0
    while i < len(z):
        j = i
        while j+1 < len(z) and zs[j+1] == zs[i]:
            j += 1
        r[o[i:j+1]] = 0.5*(i+j+2); i = j+1
    n1 = y.sum(); n0 = len(y)-n1
    return (r[y == 1].sum() - n1*(n1+1)/2)/(n1*n0)


def boot_ci(y, z, nb_=500, seed=0):
    rng = np.random.default_rng(seed); i1 = np.where(y == 1)[0]; i0 = np.where(y == 0)[0]; v = []
    for _ in range(nb_):
        k = np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]; v.append(auc(y[k], z[k]))
    return np.percentile(v, [2.5, 97.5])


def scores(d):
    """dict name -> (n, nh) chaos scores. Tangent: max over the two grid directions, and dir-0 only."""
    G, Tn, SA = d['G'], d['Tn'], d['SA']; out = {'SALI': -np.log10(np.maximum(SA, 1e-16))}
    for j, c in enumerate(TCOLS):
        out[c] = Tn[:, :, :, j].max(axis=1); out[c + '[d0]'] = Tn[:, 0, :, j]
    for j, c in enumerate(GCOLS):
        out[c] = G[:, :, j]
    return out


def analyze(files):
    res = {}
    for fn in files:
        d = np.load(fn); ref = d['ref']; H = d['H']
        lab = np.where(ref < 1e-8, 1, np.where(ref > 1e-4, 0, -1))
        fin = np.isfinite(ref) & np.all(np.isfinite(d['G']), axis=(1, 2)) & np.all(np.isfinite(d['Tn']), axis=(1, 2, 3))
        k = (lab >= 0) & fin; y = lab[k]
        sc = scores(d); tab = {name: [float(auc(y, v[k, h])) for h in range(len(H))] for name, v in sc.items()}
        res[fn] = {'H': H.tolist(), 'n': int(k.sum()), 'chaotic': float(y.mean()), 'auc': tab,
                   'h': [float(d['h1']), float(d['h2'])], 'time': float(d['time'])}
        print(f'\n{fn}: n={k.sum()} chaotic={y.mean():.2f} h=({float(d["h1"]):.1e},{float(d["h2"]):.1e})')
        print(f'{"indicator":>14s} ' + ' '.join(f'{t:6d}' for t in H))
        for name in ['SALI', 'OFLI', 'OFLI2[d0]', 'OFLI2', 'OFLI3', 'OFLI4', 'Lam2', 'Lam3', 'Lam4', 'Lammax', 'eta2', 'eta3', 'eta4',
                     'S_h', 'OFLI2_h(3pt)', 'B_h(3pt)', 'OFLI2_h', 'OFLI3_h', 'OFLI4_h', 'Lam2_h', 'Lam3_h', 'Lam4_h', 'Lammax_h',
                     'eta2_h', 'eta3_h', 'eta4_h']:
            print(f'{name:>14s} ' + ' '.join(f'{x:6.3f}' for x in tab[name]))
        ci = {}
        for name in ['SALI', 'OFLI2', 'OFLI3', 'OFLI4', 'Lam2', 'Lam4', 'Lammax', 'OFLI2_h(3pt)', 'OFLI2_h', 'OFLI3_h', 'OFLI4_h',
                     'Lam2_h', 'Lam3_h', 'Lammax_h', 'eta4_h']:
            ci[name] = [boot_ci(y, sc[name][k, h]).tolist() for h in range(len(H))]
        res[fn]['ci'] = ci
    return res


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'verify':
        verify()
    elif cmd == 'run':
        sysname, plane, E, N, n, fn = sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
        kind = KIND[sysname]; om, dt, tau, H, W0 = setup(sysname); hidx = (H/(tau*dt) - 1).round().astype(np.int64)
        S0, Z0, h1, h2 = sample(sysname, plane, E, N, n); t0 = time.time()
        G, Tn, SA, ref = run_all(kind, S0, Z0, W0, h1, h2, om, dt, tau, int(hidx[-1]) + 1, hidx, int(round(1e4/(tau*dt))))
        el = time.time() - t0
        print(fn, f'{sysname}/{plane} E={E} grid {N} h=({h1:.2e},{h2:.2e}) nodes {len(S0)} time {el:.0f}s', flush=True)
        np.savez(fn, S0=S0, G=G, Tn=Tn, SA=SA, ref=ref, H=H, h1=h1, h2=h2, E=E, time=el, sysname=sysname, plane=plane)
    elif cmd == 'analyze':
        r = analyze(sys.argv[2:])
        with open('ho_results.json', 'w') as f:
            json.dump(r, f, indent=1)
