# Figure and table scripts

Every figure of the paper has its own plotting script, which only reads `../data/` (the `paper_data/` folder of the
Zenodo archive) and never runs simulations, so it can be edited and re-run in seconds. Figures are written to
`../figs/`, table rows to `../tables/`. Common style (fonts, colours, labels, including the name **LEON** of the
fourth-order bending indicator) is in `style.py`. In the indicator code, LEON is the column `Lam4` (tangent) and
`Lam4_h` (grid).

    python make_all.py        # all figures and table rows (plotting only)

| Paper | Plot script | Data file (in `data/`) | How the data were produced |
|---|---|---|---|
| Fig. 1 aspect ratio vs SALI | `fig1_aspect_ratio.py` | `fig1_aspect_ratio.npz` | `compute_fig1_aspect_ratio.py` (~15 s) |
| Fig. 2 material contours | `fig2_contours.py` | `fig2_contours.npz` | `compute_fig2_contours.py` (~15 s) |
| Fig. 3 algorithm schematic | TikZ, `algorithm_fig3.tex` | — | drawn in LaTeX |
| Fig. 4 Hénon–Heiles AUROC | `fig3_hh_auroc.py` | `numbers.json` | `../studies/` (`gridnb.py`, `newind.py`, `fold.py`, `ucci/`; `paper_numbers*.py`) |
| Fig. 5 full-grid maps | `fig4_maps.py` | `fig4_maps.npz` | `compute_fig4_maps.py` (~5 min on 12 threads) |
| Fig. 6 grid resolution | `fig5_resolution.py` | `numbers.json` | `../studies/` (`gridnb.py`, `grid3.py`, `run_gal4.py` at several N) |
| Fig. 7 3-dof AUROC | `fig6_3dof_auroc.py` | `numbers.json` | `../studies/` (`grid3.py`, `fold3.py`) |
| Fig. 8 galactic AUROC | `fig7_galactic.py` | `numbers.json` | `../studies/` (`gal4.py`, `run_gal4.py`) |
| Fig. 9 order-4 vs order-2 gain | `fig8_order_gain.py` | `higher_order_results.json` | `../higher_order/higher_order_indicators.py run/analyze` |
| Fig. 10 robustness | `fig9_robustness.py` | `robustness.json` | `../higher_order/ho_runs/ext/run_ext.sh` + `ho_runs/robustness.py` |
| Fig. 11 parallel speed-up | `fig10_speedup.py` | `machines.json` → `speedup.json` of each machine | `compute_speedup.py` via `run_benchmarks.py` |
| Fig. 12 accuracy vs wall time | `fig11_time_to_accuracy.py` | `machines.json` → `time_to_accuracy.json` of each machine; `gpu/*/gpu.json` | `compute_time_to_accuracy.py`, `compute_gpu.py` |
| Tables V, VI | `make_tables.py` | `higher_order_results.json`, `robustness.json` | see Figs. 9–10 |
| Table VII (wall time at equal accuracy) | `make_tables.py` | `machines.json` and the files it lists | see Figs. 11–12 |
| Table VIII (GPU) | `make_tables.py` | `gpu/*/gpu.json` and the machines | `compute_gpu.py` |
| numbers quoted in Sec. VI G | `make_tables.py` | same | writes `tables/parallel_macros.tex` (`\pv{A}{key}`: i5-10400, `\pv{B}{key}`: i9-13900H, `\pv{G}{key}`: GPU) |

Plot scripts need `numpy` and `matplotlib`; compute scripts also need `numba` (and `scipy` for `../higher_order/`).

## Parallel study (Sec. VI G)

Machines are listed in `data/machines.json`. The reference machine (Intel Core i5-10400) uses `data/speedup.json`
and `data/time_to_accuracy.json`; every other machine has its own folder `data/machines/<host>/`.

To benchmark a computer (Anaconda prompt, nothing else running, laptops plugged in; `conda install tbb` gives the
threading layer used in the paper):

    python run_benchmarks.py --quick      # 2-3 min smoke test (150^2 grid), not used by the paper
    python run_benchmarks.py              # full run: ~1 h on a 6-core i5-10400, ~55 min on the i9-13900H

`run_benchmarks.py` detects the processor, physical cores and hardware threads (`machine_info.py`), chooses the
thread counts, runs

* `compute_speedup.py` — strong scaling of the SALI, tangent LEON and grid maps of Fig. 5 at T = 200
  (flat node list and row-parallel variants, repeats at 1 and all threads, check against `fig4_maps.npz`);
* `compute_time_to_accuracy.py` — AUROC versus T of every map and wall time of the SALI map versus T;
* `compute_readout_split.py` — cost of each grid indicator alone (`--only split` runs this step only);

writes `data/machines/<host>/*.json|log`, registers the machine in `data/machines.json` and regenerates Figs. 11–12,
Table VII and `tables/parallel_macros.tex`.

## GPU test

`compute_gpu.py` computes the same maps on an NVIDIA GPU (CUDA kernels compiled at run time by CuPy; one thread per
node, the same block-synchronous advance/read-out scheme, `--fmad=false` so that the orbits use the same
floating-point operations as the CPU code), in double and in single precision, on the 600^2 grid (checked against
`data/fig4_maps.npz` and by the AUROC of every map) and on larger grids (`--sizes 600,1200,2400`):

    conda install -c conda-forge cupy        # once
    python compute_gpu.py                    # -> data/gpu/<host>/gpu.json, gpu.log   (a few minutes)

`python compute_gpu.py --backend emu --sizes 120 --no-wall` compiles the same kernel source as C++/OpenMP on the CPU
(needs g++); it reproduces the CPU maps and AUROC values and was used to test the kernels without a GPU.
