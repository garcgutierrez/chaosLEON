#!/usr/bin/env python3
"""Full-cloud geometric chaos indicators for the Hénon-Heiles Poincare map.

The uncertainty distribution is represented by nested ordered probability
contours on x=0, px>0, H=E.  The contours are propagated WITHOUT Gaussian
re-seeding.  Node correspondence is retained, so stretching, non-convex
folding, curvature, and adaptive resolution growth are measurable.

Indicators per contour:
  stretch_rate = log(L/L0)/t
  UCCI         = log[(L^2/4piA)/(L0^2/4piA0)]/(2t)
  UFR          = log[(L/L_hull)/(L0/L_hull0)]/t
  curvature_rate = log[(TV/2pi)/(TV0/2pi)]/t
  refinement_rate = log(N/N0)/t

The aggregate uses probability-shell weights over nested Gaussian contours.
"""
from __future__ import annotations
import argparse, csv, json, math, time
from dataclasses import dataclass, asdict
from pathlib import Path
import matplotlib.pyplot as plt
import numba as nb
import numpy as np

E0 = 1.0 / 8.0
W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
W0 = -2.0 ** (1.0 / 3.0) / (2.0 - 2.0 ** (1.0 / 3.0))
ORBITS = {"R": 0.1, "S": -0.0135, "C": -0.015}

@dataclass
class Config:
    dt: float = 0.01
    returns: int = 80
    sigma: float = 1e-3
    probabilities: tuple = (0.2, 0.4, 0.6, 0.8, 0.95)
    initial_nodes: int = 128
    energy: float = E0
    max_return_time: float = 50.0
    adaptive: bool = True
    max_nodes: int = 4096
    refine_edge_factor: float = 4.0
    refine_angle: float = 0.35
    refinement_passes: int = 2
    scale_y: float = 1.0
    scale_py: float = 1.0

    def checked(self):
        if self.dt <= 0 or self.sigma <= 0 or self.returns < 1:
            raise ValueError("dt, sigma and returns must be positive")
        if self.initial_nodes < 16 or self.max_nodes < self.initial_nodes:
            raise ValueError("Use initial_nodes >= 16 and max_nodes >= initial_nodes")
        p = np.asarray(self.probabilities, float)
        if np.any(p <= 0) or np.any(p >= 1) or np.any(np.diff(p) <= 0):
            raise ValueError("probabilities must be strictly increasing in (0,1)")
        return self

@nb.njit(cache=True)
def H(s):
    x, y, px, py = s
    return .5*(x*x+y*y+px*px+py*py) + x*x*y - y*y*y/3.0

@nb.njit(cache=True)
def lf(s, h):
    x, y, px, py = s
    px -= .5*h*(x + 2*x*y); py -= .5*h*(y + x*x - y*y)
    x += h*px; y += h*py
    px -= .5*h*(x + 2*x*y); py -= .5*h*(y + x*x - y*y)
    s[0], s[1], s[2], s[3] = x, y, px, py

@nb.njit(cache=True)
def y4_step(s, dt):
    lf(s, W1*dt); lf(s, W0*dt); lf(s, W1*dt)

@nb.njit(cache=True)
def lf_tangent(s, V, h):
    x, y, px, py = s
    a, b, d = 1+2*y, 2*x, 1-2*y
    for j in range(2):
        V[2,j] -= .5*h*(a*V[0,j]+b*V[1,j])
        V[3,j] -= .5*h*(b*V[0,j]+d*V[1,j])
    px -= .5*h*(x+2*x*y); py -= .5*h*(y+x*x-y*y)
    x += h*px; y += h*py
    for j in range(2):
        V[0,j] += h*V[2,j]; V[1,j] += h*V[3,j]
    a, b, d = 1+2*y, 2*x, 1-2*y
    for j in range(2):
        V[2,j] -= .5*h*(a*V[0,j]+b*V[1,j])
        V[3,j] -= .5*h*(b*V[0,j]+d*V[1,j])
    px -= .5*h*(x+2*x*y); py -= .5*h*(y+x*x-y*y)
    s[0], s[1], s[2], s[3] = x, y, px, py

@nb.njit(cache=True)
def y4_tangent_step(s, V, dt):
    lf_tangent(s,V,W1*dt); lf_tangent(s,V,W0*dt); lf_tangent(s,V,W1*dt)

@nb.njit(cache=True)
def positive_px(y, py, energy):
    v = 2*energy - py*py - y*y + 2*y*y*y/3.0
    return math.sqrt(v) if v >= 0 else math.nan

@nb.njit(cache=True)
def first_return(s, dt, max_steps, energy):
    """Next x=0, px>0 crossing; linear event interpolation + energy repair."""
    seen_negative = False
    elapsed = 0.0
    for _ in range(max_steps):
        old = s.copy()
        y4_step(s, dt); elapsed += dt
        if s[0] < -1e-10:
            seen_negative = True
        if seen_negative and old[0] < 0.0 and s[0] >= 0.0 and s[2] > 0.0:
            den = s[0] - old[0]
            a = -old[0]/den if abs(den) > 1e-30 else 1.0
            y = old[1] + a*(s[1]-old[1]); py = old[3] + a*(s[3]-old[3])
            px = positive_px(y, py, energy)
            if not np.isfinite(px):
                return elapsed, False
            s[0], s[1], s[2], s[3] = 0.0, y, px, py
            return elapsed-dt+a*dt, True
    return elapsed, False

@nb.njit(cache=True, parallel=True)
def return_batch(X, dt, max_steps, energy):
    times = np.empty(X.shape[0]); ok = np.empty(X.shape[0], np.bool_)
    for i in nb.prange(X.shape[0]):
        times[i], ok[i] = first_return(X[i], dt, max_steps, energy)
    return times, ok

@nb.njit(cache=True)
def sali_value(V):
    a, b = V[:,0].copy(), V[:,1].copy()
    a /= math.sqrt(np.dot(a,a)); b /= math.sqrt(np.dot(b,b))
    return min(math.sqrt(np.sum((a+b)**2)), math.sqrt(np.sum((a-b)**2)))

@nb.njit(cache=True)
def sali_at_times(x0, targets, dt):
    s=x0.copy(); V=np.zeros((4,2)); V[0,0]=1.; V[2,1]=1.
    out=np.empty(len(targets)); t=0.; k=0
    while k < len(targets):
        y4_tangent_step(s,V,dt); t += dt
        for j in range(2):
            n=math.sqrt(np.sum(V[:,j]**2)); V[:,j] /= n
        while k < len(targets) and t >= targets[k]-0.5*dt:
            out[k]=sali_value(V); k += 1
    return out

def state_on_section(y, py, energy):
    px = positive_px(float(y), float(py), float(energy))
    if not np.isfinite(px): raise ValueError(f"Forbidden section point y={y}, py={py}")
    return np.array((0.,y,px,py),float)

def initial_contour(center_y, center_py, radius, n, cfg):
    theta=np.linspace(0,2*np.pi,n,endpoint=False)
    y=center_y+cfg.sigma*radius*cfg.scale_y*np.cos(theta)
    py=center_py+cfg.sigma*radius*cfg.scale_py*np.sin(theta)
    X=np.empty((n,4))
    for i in range(n): X[i]=state_on_section(y[i],py[i],cfg.energy)
    return theta, X

def xy(X,cfg):
    return np.column_stack((X[:,1]/cfg.scale_y, X[:,3]/cfg.scale_py))

def perimeter(P):
    return float(np.linalg.norm(np.roll(P,-1,axis=0)-P,axis=1).sum())

def polygon_area(P):
    x,y=P[:,0],P[:,1]
    return .5*abs(float(np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1))))

def convex_hull(P):
    q=sorted(set(map(tuple,P)))
    if len(q)<=1: return np.asarray(q,float)
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lo=[]
    for p in q:
        while len(lo)>=2 and cross(lo[-2],lo[-1],p)<=0: lo.pop()
        lo.append(p)
    up=[]
    for p in reversed(q):
        while len(up)>=2 and cross(up[-2],up[-1],p)<=0: up.pop()
        up.append(p)
    return np.asarray(lo[:-1]+up[:-1],float)

def absolute_turning(P):
    prev=P-np.roll(P,1,axis=0); nxt=np.roll(P,-1,axis=0)-P
    cr=prev[:,0]*nxt[:,1]-prev[:,1]*nxt[:,0]
    dot=np.sum(prev*nxt,axis=1)
    return float(np.abs(np.arctan2(cr,dot)).sum())

def geometry(X,cfg):
    P=xy(X,cfg); L=perimeter(P); A=max(polygon_area(P),1e-300)
    hull=convex_hull(P); Lh=max(perimeter(hull),1e-300)
    return dict(length=L, area=A, hull_length=Lh, hull_ratio=L/Lh,
                isoperimetric=L*L/(4*np.pi*A), curvature=absolute_turning(P)/(2*np.pi),
                nodes=len(P))

def candidate_edges(X, initial_mean_edge, cfg):
    P=xy(X,cfg); edge=np.linalg.norm(np.roll(P,-1,axis=0)-P,axis=1)
    prev=P-np.roll(P,1,axis=0); nxt=np.roll(P,-1,axis=0)-P
    ang=np.abs(np.arctan2(prev[:,0]*nxt[:,1]-prev[:,1]*nxt[:,0],np.sum(prev*nxt,axis=1)))
    flag=edge > cfg.refine_edge_factor*initial_mean_edge
    sharp=np.where(ang>cfg.refine_angle)[0]
    for i in sharp: flag[i]=True; flag[(i-1)%len(flag)]=True
    return np.where(flag)[0]

def midpoint_angles(theta, edges):
    ans=[]
    n=len(theta)
    for i in edges:
        a=theta[i]; b=theta[(i+1)%n]
        if i==n-1: b += 2*np.pi
        ans.append((.5*(a+b))%(2*np.pi))
    return np.asarray(ans)

def propagate_from_initial(center_y,center_py,radius,thetas,nreturn,cfg):
    X=np.empty((len(thetas),4))
    for i,a in enumerate(thetas):
        y=center_y+cfg.sigma*radius*cfg.scale_y*np.cos(a)
        py=center_py+cfg.sigma*radius*cfg.scale_py*np.sin(a)
        X[i]=state_on_section(y,py,cfg.energy)
    for _ in range(nreturn):
        _,ok=return_batch(X,cfg.dt,int(cfg.max_return_time/cfg.dt),cfg.energy)
        if not np.all(ok): raise RuntimeError("A refined node failed to return")
    return X

def refine(theta,X,center_y,center_py,radius,nreturn,initial_edge,cfg):
    if not cfg.adaptive: return theta,X
    for _ in range(cfg.refinement_passes):
        edges=candidate_edges(X,initial_edge,cfg)
        room=cfg.max_nodes-len(theta)
        if not len(edges) or room<=0: break
        edges=edges[:room]; mids=midpoint_angles(theta,edges)
        # Guard against numerical duplicate labels.
        keep=np.array([np.min(np.abs(np.angle(np.exp(1j*(theta-a)))))>1e-13 for a in mids])
        mids=mids[keep]
        if not len(mids): break
        Xm=propagate_from_initial(center_y,center_py,radius,mids,nreturn,cfg)
        theta=np.r_[theta,mids]; X=np.vstack((X,Xm)); order=np.argsort(theta)
        theta,X=theta[order],X[order]
    return theta,X
def shell_weights(probabilities):
    p=np.asarray(probabilities,float); w=np.diff(np.r_[0.,p]); return w/w.sum()

def run_contours(y0,py0,cfg):
    cfg.checked(); radii=np.sqrt(-2*np.log(1-np.asarray(cfg.probabilities)))
    weights=shell_weights(cfg.probabilities); contours=[]
    for r in radii:
        th,X=initial_contour(y0,py0,r,cfg.initial_nodes,cfg)
        g0=geometry(X,cfg); contours.append(dict(r=r,theta=th,X=X,g0=g0,
            initial_edge=g0["length"]/cfg.initial_nodes,elapsed=0.))
    fields=("return","time","stretch_rate","ucci","ufr","curvature_rate",
            "refinement_rate","mean_hull_ratio","mean_nodes","max_nodes","sali")
    hist=[]; xref=state_on_section(y0,py0,cfg.energy); tref=0.; reference_times=[]
    contour_rows=[]
    for nret in range(1,cfg.returns+1):
        tr,ok=first_return(xref,cfg.dt,int(cfg.max_return_time/cfg.dt),cfg.energy)
        if not ok: raise RuntimeError("Reference orbit did not return")
        tref += tr; reference_times.append(tref)
        per=[]
        for j,c in enumerate(contours):
            times,ok=return_batch(c["X"],cfg.dt,int(cfg.max_return_time/cfg.dt),cfg.energy)
            if not np.all(ok): raise RuntimeError(f"Contour {j} has non-returning node")
            c["elapsed"] += float(np.mean(times))
            c["theta"],c["X"]=refine(c["theta"],c["X"],y0,py0,c["r"],nret,c["initial_edge"],cfg)
            g=geometry(c["X"],cfg); g0=c["g0"]; t=max(c["elapsed"],1e-300)
            q=dict(stretch_rate=math.log(g["length"]/g0["length"])/t,
                   ucci=.5*math.log(g["isoperimetric"]/g0["isoperimetric"])/t,
                   ufr=math.log(g["hull_ratio"]/g0["hull_ratio"])/t,
                   curvature_rate=math.log(max(g["curvature"],1e-300)/max(g0["curvature"],1e-300))/t,
                   refinement_rate=math.log(g["nodes"]/g0["nodes"])/t,**g)
            per.append(q); contour_rows.append({"return":nret,"probability":cfg.probabilities[j],"radius":c["r"],"time":t,**q})
        agg={k:float(sum(weights[j]*per[j][k] for j in range(len(per)))) for k in
             ("stretch_rate","ucci","ufr","curvature_rate","refinement_rate","hull_ratio","nodes")}
        hist.append(dict(return_=nret,time=tref,stretch_rate=agg["stretch_rate"],ucci=agg["ucci"],ufr=agg["ufr"],
            curvature_rate=agg["curvature_rate"],refinement_rate=agg["refinement_rate"],mean_hull_ratio=agg["hull_ratio"],
            mean_nodes=agg["nodes"],max_nodes=max(q["nodes"] for q in per),sali=np.nan))
        print(f"return {nret}/{cfg.returns}: t={tref:.2f}, UCCI={agg['ucci']:.3e}, UFR={agg['ufr']:.3e}, nodes={max(q['nodes'] for q in per)}",flush=True)
    sv=sali_at_times(state_on_section(y0,py0,cfg.energy),np.asarray(reference_times),cfg.dt)
    for r,s in zip(hist,sv): r["sali"]=s
    # Normalize the Python-safe name before CSV output.
    for r in hist: r["return"]=r.pop("return_")
    return hist,contour_rows,contours

def write_csv(path,rows):
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def plot_result(out,label,hist,contours,cfg):
    t=np.array([r["time"] for r in hist])
    fig,ax=plt.subplots(3,2,figsize=(12,11),sharex=True)
    items=(("stretch_rate","material stretching rate"),("ucci","UCCI"),("ufr","UFR: nonlinear folding"),
           ("curvature_rate","curvature-complexity rate"),("refinement_rate","refinement rate"),("sali","SALI"))
    for a,(key,title) in zip(ax.flat,items):
        v=np.array([r[key] for r in hist]);
        if key=="sali": a.semilogy(t,np.maximum(v,1e-16))
        else: a.plot(t,v)
        a.set_title(title); a.grid(alpha=.25); a.set_ylabel(key)
    ax[-1,0].set_xlabel("physical time"); ax[-1,1].set_xlabel("physical time")
    fig.suptitle(f"{label}: full-cloud material-contour diagnostics, sigma={cfg.sigma:g}")
    fig.tight_layout(); fig.savefig(out/f"{label}_indicators.png",dpi=220); plt.close(fig)
    fig,a=plt.subplots(figsize=(7,6))
    for p,c in zip(cfg.probabilities,contours):
        P=xy(c["X"],cfg); a.plot(P[:,0],P[:,1],lw=1,label=f"p={p:g}, N={len(P)}")
    a.set(xlabel="scaled y",ylabel="scaled py",title=f"{label}: contours after {cfg.returns} returns")
    a.set_aspect("equal",adjustable="box"); a.grid(alpha=.2); a.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out/f"{label}_final_contours.png",dpi=220); plt.close(fig)

def demo(out,cfg):
    combined={}
    for label,y in ORBITS.items():
        sub=out/label; sub.mkdir(exist_ok=True); print("\nORBIT",label)
        hist,detail,contours=run_contours(y,0.,cfg); write_csv(sub/"aggregate.csv",hist); write_csv(sub/"per_contour.csv",detail)
        plot_result(sub,label,hist,contours,cfg); combined[label]=hist
    fig,ax=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for label,h in combined.items():
        t=[r["time"] for r in h]
        ax[0,0].plot(t,[r["stretch_rate"] for r in h],label=label)
        ax[0,1].plot(t,[r["ucci"] for r in h],label=label)
        ax[1,0].plot(t,[r["ufr"] for r in h],label=label)
        ax[1,1].semilogy(t,np.maximum([r["sali"] for r in h],1e-16),label=label)
    for a,title in zip(ax.flat,("stretch rate","UCCI","UFR","SALI")):
        a.set_title(title);a.grid(alpha=.25);a.legend()
    ax[1,0].set_xlabel("time");ax[1,1].set_xlabel("time");fig.tight_layout();fig.savefig(out/"three_orbits_ucci.png",dpi=220);plt.close(fig)

def parse_probs(s): return tuple(float(x) for x in s.split(",") if x.strip())
def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir",default="ucci_results");p.add_argument("--orbit",choices=("R","S","C","all"),default="all")
    p.add_argument("--y",type=float);p.add_argument("--py",type=float,default=0.);p.add_argument("--dt",type=float,default=.01)
    p.add_argument("--returns",type=int,default=80);p.add_argument("--sigma",type=float,default=1e-3)
    p.add_argument("--probabilities",default="0.2,0.4,0.6,0.8,0.95");p.add_argument("--nodes",type=int,default=128)
    p.add_argument("--max-nodes",type=int,default=4096);p.add_argument("--max-return-time",type=float,default=50.)
    p.add_argument("--edge-factor",type=float,default=4.);p.add_argument("--angle",type=float,default=.35)
    p.add_argument("--refinement-passes",type=int,default=2);p.add_argument("--no-adaptive",action="store_true")
    a=p.parse_args();cfg=Config(dt=a.dt,returns=a.returns,sigma=a.sigma,probabilities=parse_probs(a.probabilities),
        initial_nodes=a.nodes,max_nodes=a.max_nodes,max_return_time=a.max_return_time,refine_edge_factor=a.edge_factor,
        refine_angle=a.angle,refinement_passes=a.refinement_passes,adaptive=not a.no_adaptive).checked()
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);(out/"config.json").write_text(json.dumps(asdict(cfg),indent=2))
    t=time.perf_counter()
    if a.y is not None:
        h,d,c=run_contours(a.y,a.py,cfg);write_csv(out/"aggregate.csv",h);write_csv(out/"per_contour.csv",d);plot_result(out,"custom",h,c,cfg)
    elif a.orbit=="all": demo(out,cfg)
    else:
        h,d,c=run_contours(ORBITS[a.orbit],0.,cfg);write_csv(out/"aggregate.csv",h);write_csv(out/"per_contour.csv",d);plot_result(out,a.orbit,h,c,cfg)
    print(f"Done in {time.perf_counter()-t:.2f}s: {out.resolve()}")
if __name__=="__main__":main()
