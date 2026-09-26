#!/bin/bash
# Extension runs: more energies, second IC plane per system, fourth system (coupled quartic oscillators).
V=${PY:-python3}
cd "$(dirname "$0")/../.."
while read sys plane E n out; do
  [ -z "$sys" ] && continue
  $V higher_order_indicators.py run $sys $plane $E 1000 $n ho_runs/ext/$out
done <<LIST
hh x0 0.0833333333 2000 hh_x0_0083.npz
hh x0 0.14 2000 hh_x0_0140.npz
hh y0 0.125 2000 hh_y0_0125.npz
qo x0 0.75 2000 qo_x0_075.npz
qo x0 1.0 2000 qo_x0_100.npz
qo x0 2.0 2000 qo_x0_200.npz
3dof q0 0.08 1500 3dof_q0_008.npz
3dof p0 0.07 1500 3dof_p0_007.npz
3dof p0 0.09 1500 3dof_p0_009.npz
gal rpr -0.75 2000 gal_rpr_075.npz
gal rz -0.80 2000 gal_rz_080.npz
gal rz -0.85 2000 gal_rz_085.npz
gal rz -0.77 2000 gal_rz_077.npz
gal rz -0.745 2000 gal_rz_0745.npz
LIST
