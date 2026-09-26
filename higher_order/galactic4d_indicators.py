"""4D (2-dof) meridional galactic-type system used in the authors' earlier work:
H = (p_r^2 + p_z^2)/2 + V(r,z),  V = 0.5 ln(r^2+z^2) + a/(2 r^2) - z,  a = 0.05831, E = -0.75.
Orbit, tangent (k vectors), second variation, grid-neighbour kernels; Yoshida-4 leapfrog."""
import numpy as np, numba as nb, math, sys, time

A = 0.05831; E0 = -0.75
Y1 = 1.0/(2.0-2.0**(1.0/3.0)); Y0 = -2.0**(1.0/3.0)/(2.0-2.0**(1.0/3.0))

@nb.njit(cache=True, inline='always')
def grad(r, z):
    r2 = r*r + z*z
    return r/r2 - A/(r*r*r), z/r2 - 1.0

@nb.njit(cache=True, inline='always')
def hess(r, z):
    r2 = r*r + z*z; r4 = r2*r2
    return (z*z - r*r)/r4 + 3*A/(r**4), -2*r*z/r4, (r*r - z*z)/r4

@nb.njit(cache=True, inline='always')
def d3(r, z):
    r2 = r*r + z*z; r6 = r2*r2*r2
    vrrr = (2*r**3 - 6*r*z*z)/r6 - 12*A/r**5
    vrrz = (6*z*r*r - 2*z**3)/r6
    vrzz = -(2*r**3 - 6*r*z*z)/r6
    vzzz = (2*z**3 - 6*z*r*r)/r6
    return vrrr, vrrz, vrzz, vzzz

@nb.njit(cache=True)
def Hval(s):
    r, z = s[0], s[1]
    return 0.5*(s[2]**2 + s[3]**2) + 0.5*math.log(r*r + z*z) + A/(2*r*r) - z

def pr_from(r, z, E=E0):
    v = 2*(E - 0.5*np.log(r*r + z*z) - A/(2*r*r) + z)
    return np.sqrt(v) if (r > 0 and v >= 0) else np.nan

@nb.njit(cache=True)
def step_orbits(S, dt, n):
    for _ in range(n):
        for c in (Y1, Y0, Y1):
            h = c*dt
            for k in range(S.shape[0]):
                gr, gz = grad(S[k, 0], S[k, 1]); S[k, 2] -= 0.5*h*gr; S[k, 3] -= 0.5*h*gz
                S[k, 0] += h*S[k, 2]; S[k, 1] += h*S[k, 3]
                gr, gz = grad(S[k, 0], S[k, 1]); S[k, 2] -= 0.5*h*gr; S[k, 3] -= 0.5*h*gz

@nb.njit(cache=True)
def step_tan(s, W, dt, n):
    k = W.shape[1]
    for _ in range(n):
        for c in (Y1, Y0, Y1):
            h = c*dt
            for half in range(2):
                a, b, d = hess(s[0], s[1]); gr, gz = grad(s[0], s[1])
                for j in range(k):
                    W[2, j] -= 0.5*h*(a*W[0, j] + b*W[1, j]); W[3, j] -= 0.5*h*(b*W[0, j] + d*W[1, j])
                s[2] -= 0.5*h*gr; s[3] -= 0.5*h*gz
                if half == 0:
                    s[0] += h*s[2]; s[1] += h*s[3]
                    for j in range(k):
                        W[0, j] += h*W[2, j]; W[1, j] += h*W[3, j]

@nb.njit(cache=True)
def step2(s, w, v, dt, n):
    for _ in range(n):
        for c in (Y1, Y0, Y1):
            h = c*dt
            for half in range(2):
                r, z = s[0], s[1]
                a, b, d = hess(r, z); gr, gz = grad(r, z); vrrr, vrrz, vrzz, vzzz = d3(r, z)
                wr, wz = w[0], w[1]
                q_r = vrrr*wr*wr + 2*vrrz*wr*wz + vrzz*wz*wz
                q_z = vrrz*wr*wr + 2*vrzz*wr*wz + vzzz*wz*wz
                v[2] -= 0.5*h*(a*v[0] + b*v[1] + q_r); v[3] -= 0.5*h*(b*v[0] + d*v[1] + q_z)
                w[2] -= 0.5*h*(a*wr + b*wz); w[3] -= 0.5*h*(b*wr + d*wz)
                s[2] -= 0.5*h*gr; s[3] -= 0.5*h*gz
                if half == 0:
                    s[0] += h*s[2]; s[1] += h*s[3]
                    w[0] += h*w[2]; w[1] += h*w[3]; v[0] += h*v[2]; v[1] += h*v[3]

@nb.njit(cache=True)
def proj(s, x):
    gr, gz = grad(s[0], s[1])
    g = np.array([gr, gz, s[2], s[3]]); g /= np.sqrt(np.sum(g*g))
    f = np.array([s[2], s[3], -gr, -gz]); f -= np.dot(f, g)*g; f /= np.sqrt(np.sum(f*f))
    return x - np.dot(x, g)*g - np.dot(x, f)*f

@nb.njit(cache=True)
def sp(a, b):
    a = a/np.sqrt(np.sum(a*a)); b = b/np.sqrt(np.sum(b*b))
    return min(np.sqrt(np.sum((a+b)**2)), np.sqrt(np.sum((a-b)**2)))

@nb.njit(cache=True)
def wedge(a, b):
    na = np.sqrt(np.sum(a*a)); nb_ = np.sqrt(np.sum(b*b)); c = np.dot(a, b)/(na*nb_)
    return na*nb_*math.sqrt(max(0.0, 1 - c*c))

@nb.njit(cache=True)
def engine(x0, dt, tau, nbk, hidx):
    """cols: SALI, OFLI, MEGNO, GALI3, OFLI2, B (bending)"""
    s = x0.copy(); W = np.zeros((4, 2)); W[0, 0] = 1.0; W[1, 1] = 1.0
    s3 = x0.copy(); G = np.array([[1., 0, 0.3], [0, 1., 0.2], [0.1, 0, 1.], [0, 0.4, 0]])
    s2 = x0.copy(); w = proj(s2, np.array([1., 0, 0, 0])); w /= np.sqrt(np.sum(w*w)); v = np.zeros(4)
    out = np.full((hidx.size, 6), np.nan)
    L0 = 0.0; ofli = -1e300; lw = 0.0; S = 0.0; Yint = 0.0; t = 0.0; Lc = 0.0; ofli2 = -1e300; bend = -1e300; hk = 0
    for b in range(nbk):
        for k in range(tau):
            step_tan(s, W, dt, 1); t += dt
            ln = L0 + 0.5*math.log(np.sum(W[:, 0]**2)); S += t*(ln - lw); lw = ln; Yint += (2.0*S/t)*dt
        step_tan(s3, G, dt, tau); step2(s2, w, v, dt, tau)
        pa = proj(s, W[:, 0].copy()); n0 = math.sqrt(np.sum(W[:, 0]**2)); n1 = math.sqrt(np.sum(W[:, 1]**2))
        ofli = max(ofli, L0 + math.log(math.sqrt(np.sum(pa*pa))))
        pw = proj(s2, w); pv = proj(s2, v); npw = math.sqrt(np.sum(pw*pw))
        bend = max(bend, Lc + math.log(max(wedge(pw, pv), 1e-300)) - 2*math.log(npw))
        zz = pw + 0.5*math.exp(min(Lc, 600.0))*pv; ofli2 = max(ofli2, Lc + math.log(math.sqrt(np.sum(zz*zz))))
        for j in range(3):
            G[:, j] /= np.sqrt(np.sum(G[:, j]**2))
        if hk < hidx.size and b == hidx[hk]:
            out[hk, 0] = sp(W[:, 0], W[:, 1]); out[hk, 1] = ofli; out[hk, 2] = Yint/t
            out[hk, 3] = np.prod(np.linalg.svd(G)[1]); out[hk, 4] = ofli2; out[hk, 5] = bend; hk += 1
        L0 += math.log(n0); W[:, 0] /= n0; W[:, 1] /= n1
        nn = math.sqrt(np.sum(w*w)); Lc += math.log(nn); w /= nn; v /= nn*nn
    return out

@nb.njit(cache=True)
def sali_ref(x0, dt, tau, nbk):
    s = x0.copy(); W = np.zeros((4, 2)); W[0, 0] = 1.0; W[1, 1] = 1.0; val = 1.0
    for b in range(nbk):
        step_tan(s, W, dt, tau); val = sp(W[:, 0], W[:, 1])
        W[:, 0] /= np.sqrt(np.sum(W[:, 0]**2)); W[:, 1] /= np.sqrt(np.sum(W[:, 1]**2))
        if val < 1e-12:
            break
    return val

@nb.njit(cache=True)
def node(S0, h1, h2, dt, tau, nbk, hidx):
    """grid kernel: centre, r+-h1, z+-h2. cols: stretch, fold B_h, grid-OFLI2, grid-OFLI2 finite-size"""
    S = S0.copy(); out = np.full((hidx.size, 4), np.nan); so = -1e300; sf = -1e300; s2 = -1e300; s2h = -1e300; hk = 0
    for b in range(nbk):
        step_orbits(S, dt, tau)
        c = S[0]
        D1 = (S[1] - S[2])/(2*h1); D2 = (S[3] - S[4])/(2*h2)
        C1 = (S[1] + S[2] - 2*c)/(h1*h1); C2 = (S[3] + S[4] - 2*c)/(h2*h2)
        a1 = proj(c, D1); a2 = proj(c, D2); n1 = math.sqrt(np.sum(a1*a1)); n2 = math.sqrt(np.sum(a2*a2))
        so = max(so, math.log(max(n1, n2)))
        sf = max(sf, math.log(max(h1*wedge(a1, proj(c, C1))/(n1*n1), h2*wedge(a2, proj(c, C2))/(n2*n2), 1e-300)))
        o1 = proj(c, D1 + 0.5*C1); o2 = proj(c, D2 + 0.5*C2)
        s2 = max(s2, math.log(max(math.sqrt(np.sum(o1*o1)), math.sqrt(np.sum(o2*o2)))))
        f1 = proj(c, D1 + 0.5*h1*C1); f2 = proj(c, D2 + 0.5*h2*C2)
        s2h = max(s2h, math.log(max(math.sqrt(np.sum(f1*f1)), math.sqrt(np.sum(f2*f2)))))
        if hk < hidx.size and b == hidx[hk]:
            out[hk, 0] = so; out[hk, 1] = sf; out[hk, 2] = s2; out[hk, 3] = s2h; hk += 1
    return out

@nb.njit(parallel=True, cache=True)
def run(S0all, h1, h2, dt, tau, nbk, hidx, nbref):
    n = S0all.shape[0]
    F = np.empty((n, hidx.size, 4)); T = np.empty((n, hidx.size, 6)); ref = np.empty(n)
    for i in nb.prange(n):
        F[i] = node(S0all[i], h1, h2, dt, tau, nbk, hidx)
        T[i] = engine(S0all[i, 0], dt, tau, nbk, hidx)
        ref[i] = sali_ref(S0all[i, 0], dt, tau, nbref)
    return F, T, ref

def sample(Ngrid, nsamp, seed=5, E=E0, rr=(0.1, 0.5), zr=(-0.1, 0.6)):
    rv = np.linspace(*rr, Ngrid); zv = np.linspace(*zr, Ngrid); h1, h2 = rv[1]-rv[0], zv[1]-zv[0]
    rng = np.random.default_rng(seed); S0 = []
    while len(S0) < nsamp:
        i, j = rng.integers(1, Ngrid-1, 2); st = []
        for r, z in ((rv[i], zv[j]), (rv[i]+h1, zv[j]), (rv[i]-h1, zv[j]), (rv[i], zv[j]+h2), (rv[i], zv[j]-h2)):
            p = pr_from(r, z, E)
            if not np.isfinite(p) or p < 1e-3: break
            st.append((r, z, p, 0.0))
        if len(st) == 5: S0.append(st)
    return np.array(S0), h1, h2


def auc(y, z):
    y = np.asarray(y); z = np.asarray(z, float); o = np.argsort(z, kind='mergesort'); r = np.empty(len(z)); zs = z[o]; i = 0
    while i < len(z):
        j = i
        while j+1 < len(z) and zs[j+1] == zs[i]: j += 1
        r[o[i:j+1]] = 0.5*(i+j+2); i = j+1
    n1 = y.sum(); n0 = len(y)-n1
    return (r[y == 1].sum() - n1*(n1+1)/2)/(n1*n0)
L = lambda v: -np.log10(np.maximum(v, 1e-16))

if __name__ == '__main__':
    # Usage: python galactic4d_indicators.py run 1000 2000 gal_1000.npz   |   python galactic4d_indicators.py analyze gal_1000.npz
    if sys.argv[1] == 'run':
        sys.argv = sys.argv[1:]
        N, n, fn = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
        H = np.array([25, 50, 100, 200, 300, 500, 1000]); dt = 0.005; tau = 1000; hidx = (H/5-1).astype(np.int64)
        S0, h1, h2 = sample(N, n); t0 = time.time()
        F, T, ref = run(S0, h1, h2, dt, tau, int(H[-1]/5), hidx, 2000)
        print(fn, f'grid {N}: h=({h1:.4f},{h2:.4f}) time {time.time()-t0:.0f}s', flush=True)
        np.savez(fn, S0=S0, F=F, T=T, ref=ref, H=H, h1=h1, h2=h2)
    else:
        sys.argv = sys.argv[1:]
        for fn in sys.argv[1:]:
            d = np.load(fn); F, T, ref, H = d['F'], d['T'], d['ref'], d['H']
            lab = np.where(ref < 1e-8, 1, np.where(ref > 1e-4, 0, -1)); k = lab >= 0; y = lab[k]
            print(f'{fn}: h=({float(d["h1"]):.1e},{float(d["h2"]):.1e}) n={k.sum()} chaotic={y.mean():.2f} excluded={(lab<0).sum()}')
            print('     T   SALI  GALI3  OFLI  MEGNO  OFLI2    B   | grid: stretch  fold  OFLI2_h  OFLI2_h(fs)')
            for h, t in enumerate(H):
                r = [auc(y, L(T[k, h, 0])), auc(y, L(T[k, h, 3])), auc(y, T[k, h, 1]), auc(y, T[k, h, 2]), auc(y, T[k, h, 4]), auc(y, T[k, h, 5]),
                     auc(y, F[k, h, 0]), auc(y, F[k, h, 1]), auc(y, F[k, h, 2]), auc(y, F[k, h, 3])]
                print(f'  {t:4d}  ' + ' '.join(f'{x:.3f}' for x in r[:6]) + ' |   ' + '   '.join(f'{x:.3f}' for x in r[6:]))
