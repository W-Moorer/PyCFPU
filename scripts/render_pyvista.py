import numpy as np
import pyvista as pv
from pycfpu.cfpurecon import cfpurecon
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default=None)
    ap.add_argument('--m', type=int, default=400)
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
    potential, X, Y, Z = cfpurecon(points, normals, patches, m, kernel, regularization)
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
