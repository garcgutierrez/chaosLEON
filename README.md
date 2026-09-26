# Chaos as a breakdown of Gaussian uncertainty propagation — code

Code accompanying the paper

> A. García-Gutiérrez, A.Delgado, P. Barreiro, C. Rubio,
> *Chaos as a breakdown of Gaussian uncertainty propagation: from covariance aspect ratios to
> kernel-folding and grid-as-ensemble indicators*, submitted to *Chaos* (2026).

It contains the orbit and variational integrators, the chaos indicators (SALI, GALI, FLI/OFLI/OFLI2, MEGNO,
material probability contours, the bending indicators B and LEON, and the grid-as-ensemble indicators
S_h, B_h, OFLI2_h and LEON_h), the benchmark and GPU scripts, and one script per figure and table.
The numerical results (raw indicator data, figure data, timings) are archived separately on Zenodo:
**[DOI to be assigned]**.

## Contents

| Folder | What it contains | Paper |
|---|---|---|
| `paper/` | figure and table pipeline: one plotting script per figure (`figN_*.py`, reads `data/` only), `make_tables.py`, `make_all.py`; the compute scripts of Figs. 1, 2, 5 and of the parallel study (`compute_*.py`, `run_benchmarks.py`, `compute_gpu.py`); `algorithm_fig3.tex` (TikZ source of Fig. 3) | all figures, Tables V–VIII, Sec. VI G |
| `studies/` | first-round studies on the Hénon–Heiles, 3-dof and galactic systems (tangent and grid indicators on sampled grid nodes, material contours, Picard–Chebyshev test) and `paper_numbers.py` / `paper_numbers2.py`, which reduce the raw results to `numbers.json` | Figs. 4, 6–8, Tables I–IV, Sec. VI F |
| `higher_order/` | third- and fourth-order (LEON, OFLI3/4, η) tangent and grid indicators for all systems and planes (`higher_order_indicators.py`), the robustness study and cost benchmark (`ho_runs/`), and standalone per-system versions of the first-round indicators | Figs. 9–10, Tables V–VI |

Figure numbers refer to the paper; the plotting scripts keep their working names
(`fig3_hh_auroc.py` makes Fig. 4, …, `fig11_time_to_accuracy.py` makes Fig. 12; Fig. 3 is the TikZ schematic).
`paper/README.md` has the full script → figure → data table.

## Installation

Python ≥ 3.10 with NumPy, Numba, SciPy and Matplotlib:

    conda env create -f environment.yml      # or: pip install -r requirements.txt
    conda activate gaussian-breakdown-chaos

Numba's TBB threading layer (`conda install tbb`) was used for the timings.  The GPU test additionally needs an
NVIDIA GPU and CuPy (`conda install -c conda-forge cupy`).  Tested with Python 3.10–3.12 and Numba 0.61–0.67
on Windows 11 and Linux.

## Reproducing the figures and tables (minutes)

Download the Zenodo archive and copy its `paper_data/` folder to `data/` at the top of this repository:

    gaussian-breakdown-chaos/
      data/        <- contents of paper_data/ from Zenodo
      paper/ studies/ higher_order/

then

    cd paper
    python make_all.py            # writes figs/*.pdf|png and tables/*.tex at the top of the repository

The plotting scripts only read `data/`; they never run simulations.

## Reproducing the data (hours)

* **Figs. 1, 2, 5** (`paper/compute_fig1_aspect_ratio.py`, `compute_fig2_contours.py`, `compute_fig4_maps.py`):
  run from `paper/`, write to `data/`.
* **Parallel study, Figs. 11–12, Tables VII–VIII**: `python paper/run_benchmarks.py` on each computer
  (strong scaling, wall time versus T, cost of each grid indicator; registers the machine in
  `data/machines.json`) and `python paper/compute_gpu.py` on a CUDA GPU.  See `paper/README.md`.
* **Figs. 4, 6–8 and Tables I–IV** (`numbers.json`): see `studies/README.md`
  (`run_studies.sh` lists the commands; the raw outputs are in the Zenodo archive under `raw/studies/`).
* **Figs. 9–10, Tables V–VI** (`higher_order_results.json`, `robustness.json`): see `higher_order/README.md`
  (raw outputs under `raw/higher_order/`).

All integrations use the fourth-order Yoshida composition of the leapfrog (Δt = 0.01, Δt = 0.005 for the
galactic potential); the variational equations are the exact discrete variations of the same scheme.
Reference labels are SALI(t = 10⁴) (< 10⁻⁸ chaotic, > 10⁻⁴ regular, otherwise excluded).

## Citation

If you use this code, please cite the paper and the Zenodo record (see `CITATION.cff`).

## License

MIT (see `LICENSE`).  The data on Zenodo are distributed under CC BY 4.0.
