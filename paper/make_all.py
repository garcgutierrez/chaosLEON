"""Regenerate every figure and table of the paper from ../data (plotting only, no simulations).
Usage:  python make_all.py            (run from any directory)"""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
for s in ['fig1_aspect_ratio', 'fig2_contours', 'fig3_hh_auroc', 'fig4_maps', 'fig5_resolution', 'fig6_3dof_auroc',
          'fig7_galactic', 'fig8_order_gain', 'fig9_robustness', 'fig10_speedup', 'fig11_time_to_accuracy', 'make_tables']:
    print('--', s); subprocess.run([sys.executable, os.path.join(HERE, s + '.py')], check=True, cwd=HERE)
