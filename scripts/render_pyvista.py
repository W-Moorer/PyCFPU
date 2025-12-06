import numpy as np
import pyvista as pv
from pycfpu.cfpurecon import cfpurecon
import argparse
import time
from datetime import datetime
from pathlib import Path
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default=None)
    ap.add_argument('--m', type=int, default=400)
    ap.add_argument('--jobs', type=int, default=0)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    data_dir = root / 'data'
    if args.model:
        points = np.loadtxt(str(data_dir / f'demo_nodes__{args.model}.txt'))
        normals = np.loadtxt(str(data_dir / f'demo_normals__{args.model}.txt'))
        patches = np.loadtxt(str(data_dir / f'demo_patches__{args.model}.txt'))
    else:
        points = np.loadtxt(str(data_dir / 'demo_nodes.txt'))
        normals = np.loadtxt(str(data_dir / 'demo_normals.txt'))
        patches = np.loadtxt(str(data_dir / 'demo_patches.txt'))
    m = args.m
    kernel = {
        'phi': lambda r: -r,
        'eta': lambda r: -r,
        'zeta': lambda r: -1.0/np.where(r==0, np.inf, r),
        'order': 1
    }
    regularization = {
        'exactinterp': 1,
        'nrmlreg': 1,
        'nrmllambda': 1e-4,
        'potreg': 0
    }
    jobs = None if args.jobs == 0 else args.jobs
    M = patches.shape[0]
    used_workers = jobs if (jobs and jobs > 0) else min(M, os.cpu_count() or 1)
    bounds_min = np.min(points, axis=0)
    bounds_max = np.max(points, axis=0)
    mode = 'auto' if jobs is None else 'manual'
    print(f"model={args.model or 'default'}")
    print(f"points_shape={points.shape}")
    print(f"normals_shape={normals.shape}")
    print(f"patches_shape={patches.shape}")
    print(f"grid_m={m}")
    print(f"threads={used_workers} mode={mode}")
    print(f"bounds_min={bounds_min}")
    print(f"bounds_max={bounds_max}")
    logs_dir = root / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    potential, X, Y, Z = cfpurecon(points, normals, patches, m, kernel, regularization, jobs)
    t1 = time.perf_counter()
    log_path = logs_dir / f"render__{args.model or 'default'}__m{m}.log"
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now().isoformat()}] model={args.model or 'default'} m={m} threads={used_workers} mode={mode}\n")
        f.write(f"points_shape={points.shape} normals_shape={normals.shape} patches_shape={patches.shape}\n")
        f.write(f"bounds_min={bounds_min.tolist()} bounds_max={bounds_max.tolist()}\n")
        f.write(f"cfpurecon_time_sec={t1 - t0:.6f}\n")
    print(f"grid_shape={X.shape}")
    sg = pv.StructuredGrid(X, Y, Z)
    sg['potential'] = potential.ravel(order='F')
    iso = sg.contour(isosurfaces=[0.0])
    plotter = pv.Plotter(shape=(1, 2))
    plotter.subplot(0, 0)
    plotter.add_mesh(iso, color='lightgray', specular=0.1, smooth_shading=True, opacity=0.85)
    plotter.add_axes()
    plotter.subplot(0, 1)
    plotter.add_mesh(iso, color='lightgray', specular=0.1, smooth_shading=True, opacity=0.85)
    plotter.add_points(patches, color='red', render_points_as_spheres=True, point_size=10)
    plotter.add_axes()
    plotter.link_views()
    plotter.show()

if __name__ == '__main__':
    main()
