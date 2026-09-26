#!/bin/bash
# First-round studies: raw results (Zenodo: raw/studies/) and their reduction to numbers.json
# (Figs. 4, 6-8 and Tables I-IV of the paper).  Run from this folder; the outputs are written here.
# Timings are for a 2-core cloud machine; all scripts use every available thread (numba prange).
set -e
PY=${PY:-python}

# Henon-Heiles, E = 1/8 and 1/10, section x = 0: 2000 sampled grid nodes, each integrated with its
# four grid neighbours; tangent indicators and SALI(1e4) reference on the centre node (~30 s each)
for N in 250 500 1000 2000; do $PY gridnb.py $N 2000 gnb_$N.npz; done        # see README: gnb_* = columns 0-2
$PY gridnb.py 1000 2000 gnb_1000_E0.1.npz 0.1
$PY gridnb.py 1000 2000 gnb2_1000.npz
$PY gridnb.py 2000 2000 gnb2_2000.npz
$PY gridnb.py 1000 2000 gnb2_1000_E0.1.npz 0.1

# 3-dof Hamiltonian, plane q = 0, E = 0.06, 0.07, 0.09: 1500 sampled nodes (~1 min each)
$PY grid3.py 0.06 1000 1500 g3_06_1000.npz
$PY grid3.py 0.07 1000 1500 g3_07_1000.npz
$PY grid3.py 0.09 1000 1500 g3_09_1000.npz
$PY grid3.py 0.07  250 1500 g3_07_250.npz
$PY grid3.py 0.07 4000 1500 g3_07_4000.npz
for E in 06 07 09; do $PY grid3.py 0.$E 1000 1500 g3b_$E.npz; done          # see README: g3b_* = all columns

# galactic potential, (rho, z) plane, E = -0.75: 2000 sampled nodes (~3-4 min each)
for N in 500 1000 2000; do $PY run_gal4.py $N 2000 gal_$N.npz; done

# full 600^2 Henon-Heiles grid at T = 200: grid-as-ensemble map versus SALI map (timings of Sec. VI F)
$PY fullgrid.py 600 200                                                       # -> fullgrid_600_T200.npz

# material probability contours (UFR, UCCI) on 400 random Henon-Heiles points, sigma = 1e-4, 1e-3, 1e-2:
# 16 initial contour nodes, at most 256, 20 section returns, seed 21 (~1 s per point)
$PY ucci/ucci_auc.py 400 16 256 20 1e-3 ucci/cheap.npy 21
$PY ucci/ucci_auc.py 400 16 256 20 1e-4 ucci/cheap_s1e-4.npy 21
$PY ucci/ucci_auc.py 400 16 256 20 1e-2 ucci/cheap_s1e-2.npy 21

# reduction to numbers.json (AUROC, bootstrap confidence intervals, single-thread costs)
mkdir -p paper
$PY paper_numbers.py        # tangent indicators on the sampled nodes, UFR sets, 3-dof, costs -> paper/numbers.json
$PY paper_numbers2.py       # grid OFLI2, galactic system                                    -> paper/numbers.json
