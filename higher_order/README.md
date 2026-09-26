# Higher-order indicators and robustness study

`higher_order_indicators.py` implements, for all four systems (Hénon–Heiles `hh`, coupled quartic oscillators `qo`,
the 3-dof Hamiltonian `3dof` and the galactic potential `gal`) and their initial-condition planes, the exact
discrete first to fourth variations of the Yoshida-4 leapfrog and, from one run per sampled grid node:

* tangent indicators: SALI, OFLI, OFLI2–OFLI4, bending indicators Λ₂–Λ₄ (Λ₄ = **LEON**), Λ_max, η₂–η₄;
* grid indicators from the nine-point stencil (exact quartic collocation): S_h, OFLI2_h (three- and five-point),
  OFLI3_h, OFLI4_h, B_h, Λ₂_h–Λ₄_h (Λ₄_h = LEON_h), η₂_h–η₄_h;
* the SALI(10⁴) reference label of every node.

The docstring at the top of the file documents the definitions and the command line:

    python higher_order_indicators.py verify                                         # jet / variational checks
    python higher_order_indicators.py run hh x0 0.125 1000 2000 ho_runs/ho_hh_0125.npz  # system plane E N nodes out
    cd ho_runs && python ../higher_order_indicators.py analyze ho_*.npz             # -> ho_results.json

| Paper | Commands | Output (Zenodo `raw/higher_order/`) |
|---|---|---|
| Fig. 9, Table V (six original cases) | `run hh x0 0.125 1000 2000`, `run hh x0 0.1 1000 2000`, `run 3dof q0 0.06/0.07/0.09 1000 1500`, `run gal rz -0.75 1000 2000`; then `analyze` in `ho_runs/` | `ho_*.npz`, `ho_results.json` (= `higher_order_results.json` of `paper_data/`) |
| Fig. 10, Table VI (19 cases) | `ho_runs/ext/run_ext.sh` (14 additional cases); then `cd ho_runs && python robustness.py` | `ext/*.npz`, `robustness.json` |
| paired AUROC differences between indicators (original six cases) | `cd ho_runs && python paired.py` | printed |
| single-thread cost per unit time of each indicator | `cd ho_runs && python bench_cost.py` | `bench_cost.json` |

The run time of each case is recorded in `run.log` and `ext/run.log` of the archive (1–2 min per Hénon–Heiles or
3-dof case and about 9 min per galactic case on the machine used).

The other modules in this folder are standalone, documented versions of the first-round indicators of
`../studies/` (per system: `new_indicators_hh.py`, `fold_indicator_hh.py`, `grid_nodes_hh.py`,
`grid_ensemble_hh.py`, `uali_henon_heiles.py`; `csali_3dof.py`, `pci_3dof.py`, `grid_ensemble_3dof.py`;
`galactic4d_indicators.py`; the material-contour test `ucci_auroc_test.py` with `ucci_hh.py`). Each has its usage in
its docstring. `ho_runs/bench_cost.py` uses them to time the paper's hand-coded kernels.
