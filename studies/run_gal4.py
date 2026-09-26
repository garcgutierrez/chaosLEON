import numpy as np, sys, time
from gal4 import sample, run
N, n, fn = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
H = np.array([25, 50, 100, 200, 300, 500, 1000]); dt = 0.005; tau = 1000; hidx = (H/5-1).astype(np.int64)
S0, h1, h2 = sample(N, n); t0 = time.time()
F, T, ref = run(S0, h1, h2, dt, tau, int(H[-1]/5), hidx, 2000)
print(fn, f'grid {N}: h=({h1:.4f},{h2:.4f}) time {time.time()-t0:.0f}s', flush=True)
np.savez(fn, S0=S0, F=F, T=T, ref=ref, H=H, h1=h1, h2=h2)
