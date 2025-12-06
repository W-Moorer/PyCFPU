import numpy as np
import pyvista as pv
from pycfpu.cfpurecon import cfpurecon
from pathlib import Path
import argparse
import os
from PIL import Image
import time
from datetime import datetime

def _load(data_dir: Path, model: str | None):
    if model is None:
        points = np.loadtxt(str(data_dir / 'demo_nodes.txt'))
        normals = np.loadtxt(str(data_dir / 'demo_normals.txt'))
        patches = np.loadtxt(str(data_dir / 'demo_patches.txt'))
    else:
        points = np.loadtxt(str(data_dir / f'demo_nodes__{model}.txt'))
        normals = np.loadtxt(str(data_dir / f'demo_normals__{model}.txt'))
        patches = np.loadtxt(str(data_dir / f'demo_patches__{model}.txt'))
    return points, normals, patches

def _iter_models(data_dir: Path):
    yield None
    for fp in sorted((data_dir).glob('demo_nodes__*.txt')):
        name = fp.stem.split('__', 1)[1]
        if (data_dir / f'demo_normals__{name}.txt').exists() and (data_dir / f'demo_patches__{name}.txt').exists():
            yield name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m', type=int, default=256)
    ap.add_argument('--jobs', type=int, default=0)
    ap.add_argument('--dpi', type=int, default=600)
    ap.add_argument('--width_in', type=float, default=6.0)
    ap.add_argument('--height_in', type=float, default=4.0)
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--models', type=str, nargs='*', default=None)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    data_dir = root / 'data'
    out_dir = Path(args.out) if args.out else root / 'figs'
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = root / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f'save_all_figs__m{args.m}.log'
    jobs = None if args.jobs == 0 else args.jobs
    kernel = {'phi': lambda r: -r, 'eta': lambda r: -r, 'zeta': lambda r: -1.0/np.where(r==0, np.inf, r), 'order': 1}
    reg = {'exactinterp': 1, 'nrmlreg': 1, 'nrmllambda': 1e-4, 'potreg': 0}
    width_px = int(args.width_in * args.dpi)
    height_px = int(args.height_in * args.dpi)
    if args.models:
        models = [None if m.lower() == 'default' else m for m in args.models]
    else:
        models = list(_iter_models(data_dir))
    total = len(models)
    print(f'total_models={total} m={args.m} jobs={"auto" if jobs is None else jobs}')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f'[{datetime.now().isoformat()}] total_models={total} m={args.m} dpi={args.dpi} size=({width_px}x{height_px}) jobs={"auto" if jobs is None else jobs}\n')
    for i, model in enumerate(models):
        points, normals, patches = _load(data_dir, model)
        M = patches.shape[0]
        used_workers = jobs if (jobs and jobs > 0) else min(M, os.cpu_count() or 1)
        bounds_min = np.min(points, axis=0)
        bounds_max = np.max(points, axis=0)
        t0 = time.perf_counter()
        potential, X, Y, Z = cfpurecon(points, normals, patches, args.m, kernel, reg, jobs)
        t1 = time.perf_counter()
        sg = pv.StructuredGrid(X, Y, Z)
        sg['potential'] = potential.ravel(order='F')
        iso = sg.contour(isosurfaces=[0.0])
        plotter = pv.Plotter(shape=(1, 2), off_screen=True)
        plotter.subplot(0, 0)
        plotter.add_mesh(iso, color='lightgray', specular=0.1, smooth_shading=True, opacity=0.85)
        plotter.add_axes()
        plotter.subplot(0, 1)
        plotter.add_mesh(iso, color='lightgray', specular=0.1, smooth_shading=True, opacity=0.85)
        plotter.add_points(patches, color='red', render_points_as_spheres=True, point_size=10)
        plotter.add_axes()
        plotter.link_views()
        name = model if model is not None else 'default'
        outfile = out_dir / f'{name}_m{args.m}.png'
        plotter.screenshot(str(outfile), window_size=(width_px, height_px))
        plotter.close()
        im = Image.open(str(outfile))
        im.save(str(outfile), dpi=(args.dpi, args.dpi))
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f'[{datetime.now().isoformat()}] model={name} m={args.m} threads={used_workers} mode={"auto" if jobs is None else "manual"}\n')
            f.write(f'points_shape={points.shape} normals_shape={normals.shape} patches_shape={patches.shape}\n')
            f.write(f'bounds_min={bounds_min.tolist()} bounds_max={bounds_max.tolist()} grid_shape={X.shape}\n')
            f.write(f'cfpurecon_time_sec={t1 - t0:.6f} outfile={outfile.name}\n')
        name = model if model is not None else 'default'
        bar_len = 30
        filled = int(bar_len * (i + 1) / total) if total else bar_len
        bar = '#' * filled + '-' * (bar_len - filled)
        pct = int(100 * (i + 1) / total) if total else 100
        print(f'[{i+1}/{total}] {name} {bar} {pct}% -> {outfile.name}')

if __name__ == '__main__':
    main()
