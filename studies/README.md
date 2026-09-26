# First-round studies (`numbers.json`)

These are the exact modules and drivers that produced the raw results in `raw/studies/` of the Zenodo archive and,
through `paper_numbers.py` and `paper_numbers2.py`, the file `numbers.json` behind Figs. 4, 6, 7, 8, Tables I–IV
and the single-thread costs of Sec. VI F. `run_studies.sh` lists every command with its arguments.

| Module | Content |
|---|---|
| `uali.py` | `px_from` (Hénon–Heiles initial conditions on the section x = 0, used by all other modules) and an exploratory Gauss–Hermite (non-intrusive PCE) alignment index |
| `newind.py` | Hénon–Heiles tangent indicators from one run per point: SALI, section-projected SALI, FLI/OFLI, MEGNO (engine A), GALI₃ (engine B), SALI(10⁴) reference |
| `fold.py` | second variation (exact discrete variation of the Yoshida-4 leapfrog), bending indicator B and OFLI2 (tangent) |
| `gridnb.py` | grid-as-ensemble indicators on sampled grid nodes: each node is integrated with its four grid neighbours; columns of `F`: 0 stretching S_h, 1 kernel SALI, 2 bending B_h, 3 OFLI2_h, 4 finite-size OFLI2_h |
| `h3.py`, `pci3.py` | 3-dof Hamiltonian H = Σ ωᵢ(qᵢ²+pᵢ²)/2 + q₁²q₂ + q₁²q₃ (ω = 1, 1.4142, 1.7321): orbit, tangent map, SALI, GALI, section projection |
| `fold3.py` | second variation, B and OFLI2 for the 3-dof system |
| `grid3.py` | grid-as-ensemble indicators for the 3-dof system (plane q = 0), same columns as `gridnb.py` |
| `gal4.py`, `run_gal4.py` | galactic potential V = ½ ln(ρ²+z²) + a/(2ρ²) − z (a = 0.05831, E = −0.75, Δt = 0.005): tangent, second-order and grid indicators |
| `fullgrid.py` | complete 600² Hénon–Heiles grid at T = 200, grid-as-ensemble map versus SALI map (entry `fullgrid_600` of `numbers.json`) |
| `ucci/ucci_hh.py`, `ucci/ucci_auc.py` | adaptive material probability contours (UFR folding ratio, UCCI stretch rate) and their AUROC test |
| `mcpi.py` | vectorised modified Chebyshev–Picard iteration of the whole grid (Sec. VII, "Whole-grid propagation") |
| `aucnew_lib.py` | AUROC with ties, logistic helper |
| `paper_numbers.py`, `paper_numbers2.py` | reduction of the raw files to `paper/numbers.json` (AUROC per horizon, stratified bootstrap 95 % intervals with fixed seeds, costs per unit time) |

Notes.

* `gnb_*.npz` and `g3_*.npz` were written by an earlier version of `gridnb.py` and `grid3.py` that stored only
  columns 0–2 of `F`; the current versions write the same columns plus OFLI2_h and its finite-size variant
  (`gnb2_*.npz`, `g3b_*.npz`). For the files computed with both versions the shared columns, the nodes and the
  reference labels are identical. `paper_numbers.py` reads columns 0–2 of the first set, `paper_numbers2.py`
  columns 3–4 of the second.
* The single-thread costs (`cost_per_time_unit` in `numbers.json`) depend on the computer; everything else is
  deterministic. With the raw files of the Zenodo archive in this folder, `paper_numbers.py` followed by
  `paper_numbers2.py` reproduces `numbers.json` (apart from those costs).
* The later, standalone versions of several of these modules in `../higher_order/` (e.g. `grid_nodes_hh.py`,
  `grid_ensemble_3dof.py`, `galactic4d_indicators.py`) add drivers and documentation; they are used by the cost
  benchmark `../higher_order/ho_runs/bench_cost.py`.
