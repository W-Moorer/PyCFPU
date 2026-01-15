
import sys
import os
from pathlib import Path
import numpy as np
import time
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

try:
    from pycfpu.packages import cfpu
    from pycfpu.packages import fastcfpu
except ImportError:
    # Fallback for uninstalled package
    import packages.cfpu as cfpu
    import packages.fastcfpu as fastcfpu

def load_data(model_name):
    data_dir = project_root / 'data' / 'input_txt' / 'nonsmooth_geometry'
    # Check if specific folder exists, otherwise fallback to demo logic or just specific file
    # The LS output showed structure.
    # Let's try to find TruncatedRing
    
    nodes_file = data_dir / f'{model_name}_nodes.txt'
    normals_file = data_dir / f'{model_name}_normals.txt'
    patches_file = data_dir / f'{model_name}_patches.txt'
    
    if not nodes_file.exists():
        # Fallback to demo files in data/
        data_dir = project_root / 'data'
        nodes_file = data_dir / f'demo_nodes__{model_name}.txt'
        normals_file = data_dir / f'demo_normals__{model_name}.txt'
        patches_file = data_dir / f'demo_patches__{model_name}.txt'
    
    print(f"Loading {model_name} from {nodes_file}")
    points = np.loadtxt(str(nodes_file))
    normals = np.loadtxt(str(normals_file))
    patches = np.loadtxt(str(patches_file))
    return points, normals, patches

def compare_results(res_cpu, res_gpu):
    pot_cpu, X_cpu, Y_cpu, Z_cpu = res_cpu
    pot_gpu, X_gpu, Y_gpu, Z_gpu = res_gpu
    
    # Check grid consistency
    if not (np.allclose(X_cpu, X_gpu) and np.allclose(Y_cpu, Y_gpu) and np.allclose(Z_cpu, Z_gpu)):
        print("FAIL: Grid coordinates do not match!")
        return False
        
    # Compare potentials (ignoring NaNs)
    mask = ~np.isnan(pot_cpu) & ~np.isnan(pot_gpu)
    diff = np.abs(pot_cpu[mask] - pot_gpu[mask])
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")
    
    if max_diff < 1e-4:
        print("PASS: Results match within tolerance.")
        return True
    else:
        print("WARN: Results differ significantly.")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='TruncatedRing')
    parser.add_argument('--m', type=int, default=64) # Smaller grid for quick test
    parser.add_argument('--scale', type=int, default=1)
    args = parser.parse_args()
    
    points, normals, patches = load_data(args.model)
    
    # Scale up data for performance test
    if args.scale > 1:
        print(f"Scaling up model by {args.scale}x...")
        points_list = []
        normals_list = []
        patches_list = []
        
        # Create copies shifted in space
        n_copies = int(np.ceil(np.cbrt(args.scale)))
        shift_step = (np.max(points, axis=0) - np.min(points, axis=0)) * 1.5
        
        count = 0
        for i in range(n_copies):
            for j in range(n_copies):
                for k in range(n_copies):
                    if count >= args.scale: break
                    shift = np.array([i, j, k]) * shift_step
                    points_list.append(points + shift)
                    normals_list.append(normals)
                    # Patches need to be shifted too? 
                    # Patches are centers.
                    patches_list.append(patches + shift)
                    count += 1
                if count >= args.scale: break
            if count >= args.scale: break
            
        points = np.vstack(points_list)
        normals = np.vstack(normals_list)
        patches = np.vstack(patches_list)
        print(f"Total patches: {len(patches)}")

    kernel = {'phi': lambda r: -r, 'eta': lambda r: -r, 'zeta': lambda r: -1.0/np.where(r==0, np.inf, r), 'order': 1}
    # Re-enable Exact Interp to check Correction Part
    reg = {'exactinterp': 1, 'nrmlreg': 1, 'nrmllambda': 1e-4, 'potreg': 0}
    
    print("\n--- Running CPU Version ---")
    t0 = time.time()
    res_cpu = cfpu.cfpurecon(points, normals, patches, args.m, kernel, reg, n_jobs=4)
    t_cpu = time.time() - t0
    print(f"CPU Time: {t_cpu:.4f}s")
    
    print("\n--- Running GPU Version ---")
    t0 = time.time()
    res_gpu = fastcfpu.cfpurecon(points, normals, patches, args.m, kernelinfo=kernel, reginfo=reg)
    t_gpu = time.time() - t0
    print(f"GPU Time: {t_gpu:.4f}s")
    
    print(f"\nSpeedup: {t_cpu / t_gpu:.2f}x")
    
    compare_results(res_cpu, res_gpu)

if __name__ == "__main__":
    main()
