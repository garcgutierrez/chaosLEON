"""Single-thread cost per unit time and per initial condition, all in the paper-scripts' hand-coded style:
orbit, SALI (2 deviation vectors), 2nd-order jets (paper code) and 3rd/4th-order jets (hand-coded below,
checked against the generic engine of higher_order_indicators.py)."""
import sys, time, json, math, numpy as np, numba as nb; sys.path.insert(0, '..')
import higher_order_indicators as ho
import fold_indicator_hh as fh, galactic4d_indicators as g4, grid_ensemble_3dof as g3, grid_nodes_hh as gh, csali_3dof as c3, new_indicators_hh as nh
Y1, Y0, AG = ho.Y1, ho.Y0, ho.AG

@nb.njit(cache=True, inline='always')
def d2hh(p0, p1, q0, q1):
    return -2*(p0*q1 + p1*q0), -2*p0*q0 + 2*p1*q1

@nb.njit(cache=True)
def hh4(s, w, v, z3, z4, dt, n, order):
    for _ in range(n):
        for c in (Y1, Y0, Y1):
            h = c*dt
            for half in range(2):
                hk = 0.5*h; x, y = s[0], s[1]; a = -(1 + 2*y); b = -2*x; d = -(1 - 2*y)
                w0, w1, v0, v1 = w[0], w[1], v[0], v[1]
                ww = d2hh(w0, w1, w0, w1)
                if order >= 3:
                    wv = d2hh(w0, w1, v0, v1); t0, t1 = z3[0], z3[1]
                    z3[2] += hk*(a*t0 + b*t1 + 3*wv[0]); z3[3] += hk*(b*t0 + d*t1 + 3*wv[1])
                if order >= 4:
                    wz = d2hh(w0, w1, t0, t1); vv = d2hh(v0, v1, v0, v1)
                    z4[2] += hk*(a*z4[0] + b*z4[1] + 4*wz[0] + 3*vv[0]); z4[3] += hk*(b*z4[0] + d*z4[1] + 4*wz[1] + 3*vv[1])
                v[2] += hk*(a*v0 + b*v1 + ww[0]); v[3] += hk*(b*v0 + d*v1 + ww[1])
                w[2] += hk*(a*w0 + b*w1); w[3] += hk*(b*w0 + d*w1)
                s[2] -= hk*(x + 2*x*y); s[3] -= hk*(y + x*x - y*y)
                if half == 0:
                    for i in range(2):
                        s[i] += h*s[2+i]; w[i] += h*w[2+i]; v[i] += h*v[2+i]; z3[i] += h*z3[2+i]; z4[i] += h*z4[2+i]

@nb.njit(cache=True, inline='always')
def d2c(p0, p1, p2, q0, q1, q2):
    t = -2*p0*q0
    return -2*(p0*q1 + p1*q0 + p0*q2 + p2*q0), t, t

@nb.njit(cache=True)
def dof4(s, w, v, z3, z4, dt, n, om, order):
    for _ in range(n):
        for c in (Y1, Y0, Y1):
            h = c*dt
            for half in range(2):
                hk = 0.5*h; q0, q1, q2 = s[0], s[1], s[2]; A = om[0] + 2*q1 + 2*q2; B = 2*q0
                ww = d2c(w[0], w[1], w[2], w[0], w[1], w[2])
                if order >= 3:
                    wv = d2c(w[0], w[1], w[2], v[0], v[1], v[2])
                    z3[3] -= hk*(A*z3[0] + B*z3[1] + B*z3[2] - 3*wv[0]); z3[4] -= hk*(B*z3[0] + om[1]*z3[1] - 3*wv[1]); z3[5] -= hk*(B*z3[0] + om[2]*z3[2] - 3*wv[2])
                if order >= 4:
                    wz = d2c(w[0], w[1], w[2], z3[0], z3[1], z3[2]); vv = d2c(v[0], v[1], v[2], v[0], v[1], v[2])
                    z4[3] -= hk*(A*z4[0] + B*z4[1] + B*z4[2] - 4*wz[0] - 3*vv[0]); z4[4] -= hk*(B*z4[0] + om[1]*z4[1] - 4*wz[1] - 3*vv[1])
                    z4[5] -= hk*(B*z4[0] + om[2]*z4[2] - 4*wz[2] - 3*vv[2])
                v[3] -= hk*(A*v[0] + B*v[1] + B*v[2] - ww[0]); v[4] -= hk*(B*v[0] + om[1]*v[1] - ww[1]); v[5] -= hk*(B*v[0] + om[2]*v[2] - ww[2])
                w[3] -= hk*(A*w[0] + B*w[1] + B*w[2]); w[4] -= hk*(B*w[0] + om[1]*w[1]); w[5] -= hk*(B*w[0] + om[2]*w[2])
                s[3] -= hk*(om[0]*q0 + 2*q0*q1 + 2*q0*q2); s[4] -= hk*(om[1]*q1 + q0*q0); s[5] -= hk*(om[2]*q2 + q0*q0)
                if half == 0:
                    for i in range(3):
                        s[i] += h*om[i]*s[3+i]; w[i] += h*om[i]*w[3+i]; v[i] += h*om[i]*v[3+i]; z3[i] += h*om[i]*z3[3+i]; z4[i] += h*om[i]*z4[3+i]

@nb.njit(cache=True, inline='always')
def ip(b, nz):                       # Re(i^nz b)
    m = nz % 4
    return b.real if m == 0 else (-b.imag if m == 1 else (-b.real if m == 2 else b.imag))

@nb.njit(cache=True)
def gal4(s, w, v, z3, z4, dt, n, order):
    for _ in range(n):
        for c in (Y1, Y0, Y1):
            h = c*dt
            for half in range(2):
                hk = 0.5*h; r, z = s[0], s[1]; iw = 1.0/complex(r, z); i2 = iw*iw; i3 = i2*iw
                b2 = -i2; b3 = 2*i3
                T2 = (ip(b2, 0) + 3*AG/r**4, ip(b2, 1), ip(b2, 2))
                T3 = (ip(b3, 0) - 12*AG/r**5, ip(b3, 1), ip(b3, 2), ip(b3, 3))
                if order >= 3:
                    b4 = -6*i3*iw; T4 = (ip(b4, 0) + 60*AG/r**6, ip(b4, 1), ip(b4, 2), ip(b4, 3), ip(b4, 4))
                if order >= 4:
                    b5 = 24*i3*i2; T5 = (ip(b5, 0) - 360*AG/r**7, ip(b5, 1), ip(b5, 2), ip(b5, 3), ip(b5, 4), ip(b5, 5))
                w0, w1, v0, v1, t0, t1 = w[0], w[1], v[0], v[1], z3[0], z3[1]
                for i in range(2):
                    DFw = T2[i]*w0 + T2[i+1]*w1
                    D2ww = T3[i]*w0*w0 + 2*T3[i+1]*w0*w1 + T3[i+2]*w1*w1
                    DFv = T2[i]*v0 + T2[i+1]*v1
                    if order >= 3:
                        D2wv = T3[i]*w0*v0 + T3[i+1]*(w0*v1 + w1*v0) + T3[i+2]*w1*v1
                        D3www = T4[i]*w0**3 + 3*T4[i+1]*w0*w0*w1 + 3*T4[i+2]*w0*w1*w1 + T4[i+3]*w1**3
                        z3[2+i] -= hk*(T2[i]*t0 + T2[i+1]*t1 + 3*D2wv + D3www)
                    if order >= 4:
                        D2wz = T3[i]*w0*t0 + T3[i+1]*(w0*t1 + w1*t0) + T3[i+2]*w1*t1
                        D2vv = T3[i]*v0*v0 + 2*T3[i+1]*v0*v1 + T3[i+2]*v1*v1
                        D3wwv = T4[i]*w0*w0*v0 + T4[i+1]*(w0*w0*v1 + 2*w0*w1*v0) + T4[i+2]*(2*w0*w1*v1 + w1*w1*v0) + T4[i+3]*w1*w1*v1
                        D4 = T5[i]*w0**4 + 4*T5[i+1]*w0**3*w1 + 6*T5[i+2]*w0*w0*w1*w1 + 4*T5[i+3]*w0*w1**3 + T5[i+4]*w1**4
                        z4[2+i] -= hk*(T2[i]*z4[0] + T2[i+1]*z4[1] + 4*D2wz + 3*D2vv + 6*D3wwv + D4)
                    v[2+i] -= hk*(DFv + D2ww); w[2+i] -= hk*DFw
                r2 = r*r + z*z; s[2] -= hk*(r/r2 - AG/(r*r*r)); s[3] -= hk*(z/r2 - 1.0)
                if half == 0:
                    for i in range(2):
                        s[i] += h*s[2+i]; w[i] += h*w[2+i]; v[i] += h*v[2+i]; z3[i] += h*z3[2+i]; z4[i] += h*z4[2+i]

def best(f, rep=7):
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return min(ts)

res = {}
for sysname, E, (a, b) in (('hh', 0.125, (-0.015, 0.0)), ('3dof', 0.07, (0.05, 0.1)), ('gal', -0.75, (0.25, 0.2))):
    kind = ho.KIND[sysname]; om, dt, tau, _, W0 = ho.setup(sysname); x0 = ho.ic_state(sysname, E, a, b); Z0 = ho.ic_jet(sysname, E, a, b, 0)
    T = 200.0; n = int(round(T/dt))
    if sysname == 'hh':
        fast = lambda o, x, Z: hh4(x, Z[0], Z[1], Z[2], Z[3], dt, n, o)
        orbit = lambda: gh.y4n(x0[None].copy(), dt, n)
        sali = lambda: [nh.lf_tan(x, V, c*dt) for x, V in [(x0.copy(), W0.T.copy())] for _ in range(0)] or nh.engineB  # placeholder
        V2 = W0.T.copy()
        @nb.njit
        def sali_hh(x, V, dt, n):
            for _ in range(n):
                for c in (Y1, Y0, Y1):
                    nh.lf_tan(x, V, c*dt)
        sali = lambda: sali_hh(x0.copy(), V2.copy(), dt, n)
    elif sysname == '3dof':
        fast = lambda o, x, Z: dof4(x, Z[0], Z[1], Z[2], Z[3], dt, n, om, o)
        orbit = lambda: g3.y4_orbits(x0[None].copy(), dt, n, om)
        sali = lambda: c3.y4_tan(x0.copy(), W0.T.copy(), dt, n, om)
    else:
        fast = lambda o, x, Z: gal4(x, Z[0], Z[1], Z[2], Z[3], dt, n, o)
        orbit = lambda: g4.step_orbits(x0[None].copy(), dt, n)
        sali = lambda: g4.step_tan(x0.copy(), W0.T.copy(), dt, n)
    # correctness of the hand-coded order-4 step against the verified generic engine
    xa, Za = x0.copy(), Z0.copy(); fast(4, xa, Za); xb, Zb = x0.copy(), Z0.copy(); ho.step_jet(kind, xb, Zb, dt, n, om, np.zeros((6, 6)))
    rel = max(np.linalg.norm(Za[k] - Zb[k])/np.linalg.norm(Zb[k]) for k in range(4))
    tests = {'orbit': orbit, 'SALI': sali,
             'order2 (paper)': (lambda: fh.y4_2(x0.copy(), Z0[0].copy(), np.zeros(4), dt, n)) if sysname == 'hh' else
                               (lambda: g3.step2(x0.copy(), Z0[0].copy(), np.zeros(6), dt, n, om)) if sysname == '3dof' else
                               (lambda: g4.step2(x0.copy(), Z0[0].copy(), np.zeros(4), dt, n)),
             'order2': lambda: fast(2, x0.copy(), Z0.copy()), 'order3': lambda: fast(3, x0.copy(), Z0.copy()), 'order4': lambda: fast(4, x0.copy(), Z0.copy())}
    for f in tests.values():
        f()
    res[sysname] = {k: best(f)/T for k, f in tests.items()}; res[sysname]['check_rel_err'] = rel
    print(f'{sysname}: check vs generic engine rel.err {rel:.1e}')
    print('   ' + '  '.join(f'{k}: {v:.2e} ({v/res[sysname]["SALI"]:.2f}x SALI)' for k, v in res[sysname].items() if k != 'check_rel_err'))
json.dump(res, open('bench_cost.json', 'w'), indent=1)
