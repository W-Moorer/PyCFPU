# fastcfpu.py
# GPU-accelerated version of cfpu.py using CuPy with Optimized Kernels
# Created for high-performance Implicit Surface Reconstruction

import numpy as np
import cupy as cp
import cupyx
import cupyx.scipy.sparse as csp
from scipy.spatial import cKDTree
import time
import os

# --- CUDA KERNELS ---

# Kernel for Matrix Assembly (Approximation for standard kernels)
# Supporting general kernel via template might be complex, so we hardcode Order 1, eta=-r
# This replaces the huge Python broadcasting.
# Grid: (B, N, N) -> flattened or 3D grid.
# Actually, let's keep matrix assembly in CuPy for now (it's O(N^2) but N is small ~100). 
# The bottleneck is Grid Evaluation O(N * M_grid).

# Kernel for Grid Evaluation
# Arguments:
# - grid_min (3,), grid_dx (1,)
# - grid_dims (3,)
# - centers (B, 3), radii (B,)
# - x_sup (B, N_max, 3)
# - coeffs (B, 3*N_max + L) -> (B, 3, N_max) and (B, L)
# - mask (B, N_max) (implicitly 0 coeffs for invalid points)
# - global_potential (M_grid,), global_weight (M_grid,)

eval_kernel_code = r'''
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

extern "C" __global__
void eval_grid_kernel(
    const double* __restrict__ centers,      // (B, 3)
    const double* __restrict__ radii,        // (B,)
    const int*    __restrict__ min_ix_arr,   // (B,)
    const int*    __restrict__ max_ix_arr,   // (B,)
    const int*    __restrict__ min_iy_arr,   // (B,)
    const int*    __restrict__ max_iy_arr,   // (B,)
    const int*    __restrict__ min_iz_arr,   // (B,)
    const int*    __restrict__ max_iz_arr,   // (B,)
    const double* __restrict__ x_sup,        // (B, N_max, 3)
    const double* __restrict__ cx,           // (B, N_max)
    const double* __restrict__ cy,           // (B, N_max)
    const double* __restrict__ cz,           // (B, N_max)
    const double* __restrict__ cp_poly,      // (B, L) (L=3 for order 1)
    const double* __restrict__ c_corr_vec,   // (B, N_max) (Optional, can be NULL)
    const double* __restrict__ c_corr_const, // (B,) (Optional)
    const int B, const int N_max, const int L,
    const double grid_start_x, const double grid_start_y, const double grid_start_z,
    const double grid_dx,
    const int mmx, const int mmy, const int mmz,
    double* __restrict__ global_potential,
    double* __restrict__ global_weight
) {
    // Each thread processes one grid point within a patch's bounding box.
    
    int bid = blockIdx.z; // Patch Index
    if (bid >= B) return;
    
    double center_x = centers[bid * 3 + 0];
    double center_y = centers[bid * 3 + 1];
    double center_z = centers[bid * 3 + 2];
    double radius = radii[bid];
    double r2 = radius * radius;
    
    // Use pre-calculated indices from Python (matches CPU logic exactly)
    int min_ix = min_ix_arr[bid];
    int max_ix = max_ix_arr[bid];
    int min_iy = min_iy_arr[bid];
    int max_iy = max_iy_arr[bid];
    int min_iz = min_iz_arr[bid];
    int max_iz = max_iz_arr[bid];
    
    int nx = max_ix - min_ix + 1;
    int ny = max_iy - min_iy + 1;
    int nz = max_iz - min_iz + 1;
    
    if (nx <= 0 || ny <= 0 || nz <= 0) return;
    
    // Linear thread index within the box
    int tid = blockIdx.x * blockDim.x + threadIdx.x + (blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x * gridDim.x;
    
    // Map tid to (lz, ly, lx) in box
    if (tid >= nx * ny * nz) return;
    
    int lx = tid % nx;
    int rem = tid / nx;
    int ly = rem % ny;
    int lz = rem / ny;
    
    int gx = min_ix + lx;
    int gy = min_iy + ly;
    int gz = min_iz + lz;
    
    double px = grid_start_x + (gx - 1) * grid_dx;
    double py = grid_start_y + (gy - 1) * grid_dx;
    double pz = grid_start_z + (gz - 1) * grid_dx;
    
    double dx_c = px - center_x;
    double dy_c = py - center_y;
    double dz_c = pz - center_z;
    double dist2 = dx_c*dx_c + dy_c*dy_c + dz_c*dz_c;
    
    if (dist2 >= r2) return;
    
    // Evaluate Potential
    double val = 0.0;
    double dist = sqrt(dist2);
    
    for (int i = 0; i < N_max; ++i) {
        double xi = x_sup[(bid * N_max + i) * 3 + 0];
        double yi = x_sup[(bid * N_max + i) * 3 + 1];
        double zi = x_sup[(bid * N_max + i) * 3 + 2];
        
        double cxi = cx[bid * N_max + i];
        double cyi = cy[bid * N_max + i];
        double czi = cz[bid * N_max + i];
        
        if (abs(cxi) < 1e-12 && abs(cyi) < 1e-12 && abs(czi) < 1e-12) continue;
        
        double dxi = px - xi;
        double dyi = py - yi;
        double dzi = pz - zi;
        double ri = sqrt(dxi*dxi + dyi*dyi + dzi*dzi);
        
        double eta_val = -ri; // Hardcoded eta=-r
        
        val += eta_val * (dxi * cxi + dyi * cyi + dzi * czi);
    }
    
    double cpx = cp_poly[bid * L + 0];
    double cpy = cp_poly[bid * L + 1];
    double cpz = cp_poly[bid * L + 2];
    
    val += px * cpx + py * cpy + pz * cpz;
    
    // Correction Part
    if (c_corr_vec != NULL) {
        double corr = 0.0;
        for (int i = 0; i < N_max; ++i) {
            double c_val = c_corr_vec[bid * N_max + i];
            if (abs(c_val) < 1e-12) continue;
            
            double xi = x_sup[(bid * N_max + i) * 3 + 0];
            double yi = x_sup[(bid * N_max + i) * 3 + 1];
            double zi = x_sup[(bid * N_max + i) * 3 + 2];
            
            double dxi = px - xi;
            double dyi = py - yi;
            double dzi = pz - zi;
            double ri = sqrt(dxi*dxi + dyi*dyi + dzi*dzi);
            
            double phi_val = -ri;
            
            corr += phi_val * c_val;
        }
        if (c_corr_const != NULL) {
            corr += c_corr_const[bid];
        }
        val -= corr;
    }
    
    // Weight Function for blending
    double r_norm_center = dist / radius;
    double psi = 0.0;
    if (r_norm_center <= (1.0/3.0)) {
        psi = 0.75 - 2.25 * r_norm_center * r_norm_center;
    } else if (r_norm_center <= 1.0) {
        double term = (1.0 - r_norm_center);
        psi = 1.125 * term * term;
    }
    
    long long global_idx = (gy - 1) + (long long)(gx - 1) * mmy + (long long)(gz - 1) * mmx * mmy;
    
    atomicAdd(&global_potential[global_idx], val * psi);
    atomicAdd(&global_weight[global_idx], psi);
}
'''

eval_kernel = cp.RawKernel(eval_kernel_code, 'eval_grid_kernel')

class PolynomialBasis:
    @staticmethod
    def compute(x, order):
        input_shape = x.shape
        x_flat = x.reshape(-1, 3)
        n = x_flat.shape[0]
        if order == 1:
            CP = cp.zeros((3 * n, 3), dtype=x.dtype)
            P = cp.zeros((n, 3), dtype=x.dtype)
            P[:, 0:3] = x_flat
            CP_view = CP.reshape(n, 3, 3)
            CP_view[:] = cp.eye(3, dtype=x.dtype)
            return CP, P
        return None, None

class BatchLocalSolver:
    @staticmethod
    def solve_batch(x_batch, nrml_batch, mask_batch, patch_radii, kernel_info, reg_info):
        B, N, _ = x_batch.shape
        order = kernel_info.get('order', 1)
        eta = kernel_info.get('eta', lambda r: -r)
        phi = kernel_info.get('phi', lambda r: -r)
        
        nrmlreg = reg_info.get('nrmlreg', 1)
        nrmllambda = reg_info.get('nrmllambda', 1e-4)
        exactinterp = reg_info.get('exactinterp', 1)
        potlambda = reg_info.get('potlambda', 0)
        
        xx = x_batch[:, :, 0]
        xy = x_batch[:, :, 1]
        xz = x_batch[:, :, 2]
        
        dx = xx[:, :, None] - xx[:, None, :]
        dy = xy[:, :, None] - xy[:, None, :]
        dz = xz[:, :, None] - xz[:, None, :]
        
        r = cp.sqrt(dx**2 + dy**2 + dz**2)
        
        eta_r = eta(r)
        inv_r = cp.zeros_like(r)
        mask_r = r > 1e-9
        inv_r[mask_r] = 1.0 / r[mask_r]
        zeta_r = -inv_r
        
        dphi_xx = zeta_r * dx**2 + eta_r
        dphi_yy = zeta_r * dy**2 + eta_r
        dphi_zz = zeta_r * dz**2 + eta_r
        dphi_xy = zeta_r * dx * dy
        dphi_xz = zeta_r * dx * dz
        dphi_yz = zeta_r * dy * dz
        
        eye_mask = cp.tile(cp.eye(N, dtype=bool)[None, :, :], (B, 1, 1))
        dphi_xx[eye_mask] = 0
        dphi_yy[eye_mask] = 0
        dphi_zz[eye_mask] = 0
        dphi_xy[eye_mask] = 0
        dphi_xz[eye_mask] = 0
        dphi_yz[eye_mask] = 0
        
        L = 3 if order == 1 else 9
        M_sys = 3 * N + L
        A = cp.zeros((B, M_sys, M_sys), dtype=x_batch.dtype)
        
        A[:, 0:3*N:3, 0:3*N:3] = dphi_xx
        A[:, 0:3*N:3, 1:3*N:3] = dphi_xy
        A[:, 0:3*N:3, 2:3*N:3] = dphi_xz
        A[:, 1:3*N:3, 0:3*N:3] = dphi_xy
        A[:, 1:3*N:3, 1:3*N:3] = dphi_yy
        A[:, 1:3*N:3, 2:3*N:3] = dphi_yz
        A[:, 2:3*N:3, 0:3*N:3] = dphi_xz
        A[:, 2:3*N:3, 1:3*N:3] = dphi_yz
        A[:, 2:3*N:3, 2:3*N:3] = dphi_zz
        
        x_flat = x_batch.reshape(-1, 3)
        CP_flat, P_flat = PolynomialBasis.compute(x_flat, order)
        CP = CP_flat.reshape(B, 3*N, L)
        P = P_flat.reshape(B, N, L)
        
        A[:, 0:3*N, 3*N:] = CP
        A[:, 3*N:, 0:3*N] = CP.transpose(0, 2, 1)
        
        b = cp.zeros((B, M_sys), dtype=x_batch.dtype)
        b[:, 0:3*N:3] = nrml_batch[:, :, 0]
        b[:, 1:3*N:3] = nrml_batch[:, :, 1]
        b[:, 2:3*N:3] = nrml_batch[:, :, 2]
        
        # FIX: Regularization must depend on ACTUAL number of points per patch, not N_max
        n_points = cp.sum(mask_batch, axis=1) # (B,)
        reg_vals = 3 * n_points * nrmllambda # (B,)
        
        # Apply regularization to diagonal of 3N x 3N block
        # reg_vals[:, None, None] broadcasts to (B, 1, 1)
        reg_update = reg_vals[:, None, None] * cp.eye(3*N, dtype=x_batch.dtype)[None, :, :]
        A[:, 0:3*N, 0:3*N] += reg_update
        
        mask_3n = cp.repeat(mask_batch, 3, axis=1)
        m_col = mask_3n[:, :, None]
        m_row = mask_3n[:, None, :]
        
        A[:, 0:3*N, :] *= m_col
        A[:, :, 0:3*N] *= m_row
        b[:, 0:3*N] *= mask_3n
        
        diag_add = cp.eye(3*N, dtype=x_batch.dtype)[None, :, :] * (~mask_3n[:, :, None])
        A[:, 0:3*N, 0:3*N] += diag_add
        
        sol = cp.linalg.solve(A, b)
        coeffs = sol[:, :3*N]
        coeffsp = sol[:, 3*N:]
        
        coeffs_corr = None
        if exactinterp:
            cx = coeffs[:, 0:3*N:3]
            cy = coeffs[:, 1:3*N:3]
            cz = coeffs[:, 2:3*N:3]
            term = eta_r * (dx * cx[:, None, :] + dy * cy[:, None, :] + dz * cz[:, None, :])
            temp_pot = cp.sum(term, axis=2)
            poly_term = cp.matmul(P, coeffsp[:, :, None]).squeeze(-1)
            temp_pot += poly_term
            
            N1 = N + 1
            A1 = cp.zeros((B, N1, N1), dtype=x_batch.dtype)
            phi_r = phi(r)
            A1[:, :N, :N] = phi_r
            ones = cp.ones((B, N), dtype=x_batch.dtype) * mask_batch
            A1[:, :N, N] = ones
            A1[:, N, :N] = ones
            b1 = cp.zeros((B, N1), dtype=x_batch.dtype)
            b1[:, :N] = temp_pot * mask_batch
            
            # DEBUG: Print Correction System Stats
            if False: # Disable debug print
                b1_cpu = cp.asnumpy(b1)
                print(f"GPU Corr RHS (b1) Batch 0: Mean={np.mean(b1_cpu):.6e}, Max={np.max(np.abs(b1_cpu)):.6e}, AbsSum={np.sum(np.abs(b1_cpu)):.6e}")
            
            reg_pot = N * potlambda
            A1[:, :N, :N] += cp.eye(N, dtype=x_batch.dtype)[None, :, :] * reg_pot
            
            m1 = mask_batch
            m1_aug = cp.concatenate([m1, cp.ones((B, 1), dtype=bool)], axis=1)
            m1_col = m1_aug[:, :, None]
            m1_row = m1_aug[:, None, :]
            A1 *= (m1_col * m1_row)
            diag_add1 = cp.eye(N1, dtype=x_batch.dtype)[None, :, :] * (~m1_aug[:, :, None])
            A1 += diag_add1
            
            sol1 = cp.linalg.solve(A1, b1)
            coeffs_corr = sol1
            
        return coeffs, coeffsp, coeffs_corr

class CFPUSolver:
    def __init__(self, kernel_info=None, reg_info=None, n_jobs=None):
        self.kernel_info = kernel_info or {}
        self.reg_info = reg_info or {}
        self.n_jobs = n_jobs
        
    def fit(self, x, nrml):
        self.minxx = np.min(x, axis=0)
        self.maxxx = np.max(x, axis=0)
        scale = np.max(self.maxxx - self.minxx)
        self.x_norm_cpu = (x - self.minxx) / scale
        self.nrml_cpu = nrml
        self.scale = scale
        self.x_norm_gpu = cp.asarray(self.x_norm_cpu)
        self.nrml_gpu = cp.asarray(self.nrml_cpu)
        return self

    def reconstruct(self, x, nrml, centers, gridsize):
        self.fit(x, nrml)
        y_norm = (centers - self.minxx) / self.scale
        M = y_norm.shape[0]
        
        # KDTree (CPU)
        tree_y = cKDTree(y_norm)
        nn_dist = tree_y.query(y_norm, k=2)[0][:, 1]
        H = np.max(nn_dist)
        patchRad0 = (1.0 + 1.0) * H / 2.0
        
        tree_x = cKDTree(self.x_norm_cpu)
        idx = []
        for k in range(M):
            idx.append(np.array(tree_x.query_ball_point(y_norm[k, :], patchRad0), dtype=int))
            
        # Grid setup
        minx = np.min(self.x_norm_cpu, axis=0)
        maxx = np.max(self.x_norm_cpu, axis=0)
        griddx = np.max((maxx - minx) / gridsize)
        
        pad_vox = max(12, int(np.ceil(patchRad0 / griddx)) + 2)
        
        startx, starty, startz = minx - pad_vox*griddx
        
        xx = np.arange(startx, maxx[0] + pad_vox*griddx + griddx/2, griddx)
        yy = np.arange(starty, maxx[1] + pad_vox*griddx + griddx/2, griddx)
        zz = np.arange(startz, maxx[2] + pad_vox*griddx + griddx/2, griddx)
        mmx, mmy, mmz = len(xx), len(yy), len(zz)
        m = mmx * mmy * mmz
        
        Psi_sum_gpu = cp.zeros(m, dtype=cp.float64)
        potential_gpu = cp.zeros(m, dtype=cp.float64)
        
        # Batching Strategy
        lengths = np.array([len(i) for i in idx])
        perm = np.argsort(lengths)
        batch_size = 512 # Massive batch size for Kernel saturation
        
        print(f"Processing {M} patches in batches of {batch_size}...")
        
        # Pre-convert centers to GPU
        y_norm_gpu = cp.asarray(y_norm)
        
        # Kernel Constants
        order = self.kernel_info.get('order', 1)
        L = 3 if order == 1 else 9
        
        for i in range(0, M, batch_size):
            batch_indices = perm[i:min(i + batch_size, M)]
            current_batch_size = len(batch_indices)
            
            batch_lens = lengths[batch_indices]
            max_len = np.max(batch_lens)
            if max_len == 0: continue
            
            x_batch = cp.zeros((current_batch_size, max_len, 3), dtype=cp.float64)
            nrml_batch = cp.zeros((current_batch_size, max_len, 3), dtype=cp.float64)
            mask_batch = cp.zeros((current_batch_size, max_len), dtype=bool)
            
            # Fill Batch (Python loop here is still a bottleneck but smaller)
            for bi, k in enumerate(batch_indices):
                l = batch_lens[bi]
                if l > 0:
                    idk = cp.asarray(idx[k]) # Transfer indices
                    x_batch[bi, :l] = self.x_norm_gpu[idk]
                    nrml_batch[bi, :l] = self.nrml_gpu[idk]
                    mask_batch[bi, :l] = True
            
            patch_radii = cp.full(current_batch_size, patchRad0, dtype=cp.float64)
            
            # Solve System
            coeffs_batch, coeffsp_batch, coeffs_corr_batch = BatchLocalSolver.solve_batch(
                x_batch, nrml_batch, mask_batch, patch_radii, self.kernel_info, self.reg_info
            )
            
            # DEBUG: Print Coeffs Stats
            if i == 0 and False: # Disable debug
                c_cpu = cp.asnumpy(coeffs_batch)
                print(f"GPU Coeffs Batch 0: Sum={np.sum(c_cpu):.6e}, Mean={np.mean(c_cpu):.6e}, AbsSum={np.sum(np.abs(c_cpu)):.6e}")
                if coeffs_corr_batch is not None:
                    cc_cpu = cp.asnumpy(coeffs_corr_batch)
                    print(f"GPU Corr Coeffs Batch 0: Sum={np.sum(cc_cpu):.6e}")
            
            # Evaluate using Custom Kernel
            centers_batch = y_norm_gpu[batch_indices]
            
            # --- PRE-CALCULATE BOUNDS (CPU Match Logic) ---
            # CPU: ix = int(round((y0 - startx)/dx)) + 1
            #      factor = int(round(R/dx))
            #      min_ix = max(ix - factor, 1)
            #      max_ix = min(ix + factor, mmx)
            
            # Note: cp.round matches np.round (nearest even)
            ix_c = cp.round((centers_batch[:, 0] - startx) / griddx).astype(cp.int32) + 1
            iy_c = cp.round((centers_batch[:, 1] - starty) / griddx).astype(cp.int32) + 1
            iz_c = cp.round((centers_batch[:, 2] - startz) / griddx).astype(cp.int32) + 1
            
            factor = int(np.round(patchRad0 / griddx))
            
            min_ix_arr = cp.maximum(ix_c - factor, 1)
            max_ix_arr = cp.minimum(ix_c + factor, mmx)
            min_iy_arr = cp.maximum(iy_c - factor, 1)
            max_iy_arr = cp.minimum(iy_c + factor, mmy)
            min_iz_arr = cp.maximum(iz_c - factor, 1)
            max_iz_arr = cp.minimum(iz_c + factor, mmz)
            
            # Determine Grid Size for Kernel Launch
            # We need the max volume in this batch
            max_vol = cp.max((max_ix_arr - min_ix_arr + 1) * 
                             (max_iy_arr - min_iy_arr + 1) * 
                             (max_iz_arr - min_iz_arr + 1)).item()
            
            # Unpack coefficients for kernel
            # CRITICAL FIX: Ensure contiguous arrays for Kernel
            cx = cp.ascontiguousarray(coeffs_batch[:, 0:3*max_len:3], dtype=cp.float64)
            cy = cp.ascontiguousarray(coeffs_batch[:, 1:3*max_len:3], dtype=cp.float64)
            cz = cp.ascontiguousarray(coeffs_batch[:, 2:3*max_len:3], dtype=cp.float64)
            cp_poly = cp.ascontiguousarray(coeffsp_batch, dtype=cp.float64)
            
            c_corr_vec = None
            c_corr_const = None
            if coeffs_corr_batch is not None:
                c_corr_vec = cp.ascontiguousarray(coeffs_corr_batch[:, :max_len], dtype=cp.float64)
                c_corr_const = cp.ascontiguousarray(coeffs_corr_batch[:, max_len], dtype=cp.float64)
            
            threads_per_block = 256
            blocks_per_grid_x = (max_vol + threads_per_block - 1) // threads_per_block
            
            x_batch = cp.ascontiguousarray(x_batch, dtype=cp.float64)
            
            eval_kernel(
                (blocks_per_grid_x, 1, current_batch_size), (threads_per_block, 1, 1),
                (
                    centers_batch.astype(cp.float64), 
                    patch_radii, 
                    min_ix_arr, max_ix_arr,
                    min_iy_arr, max_iy_arr,
                    min_iz_arr, max_iz_arr,
                    x_batch,
                    cx, cy, cz, cp_poly,
                    c_corr_vec if c_corr_vec is not None else 0, # Handle NULL
                    c_corr_const if c_corr_const is not None else 0,
                    np.int32(current_batch_size), np.int32(max_len), np.int32(L),
                    np.float64(startx), np.float64(starty), np.float64(startz),
                    np.float64(griddx),
                    np.int32(mmx), np.int32(mmy), np.int32(mmz),
                    potential_gpu, Psi_sum_gpu
                )
            )
            
        # Finalize
        # DEBUG: Print Stats
        if False:
            psi_sum_cpu = cp.asnumpy(Psi_sum_gpu)
            pot_sum_cpu = cp.asnumpy(potential_gpu)
            print(f"GPU Psi_sum: Mean={np.mean(psi_sum_cpu):.6e}, Max={np.max(psi_sum_cpu):.6e}, Sum={np.sum(psi_sum_cpu):.6e}")
            print(f"GPU Potential Sum (Pre-Div): Mean={np.mean(pot_sum_cpu):.6e}, Max={np.max(pot_sum_cpu):.6e}, Sum={np.sum(pot_sum_cpu):.6e}")

        mask = Psi_sum_gpu > 0
        potential_gpu[mask] /= Psi_sum_gpu[mask]
        potential_gpu[~mask] = cp.nan
        
        potential = cp.asnumpy(potential_gpu)
        potential = potential.reshape((mmy, mmx, mmz), order='F')
        X, Y, Z = np.meshgrid(xx, yy, zz, indexing='xy')
        X = X * self.scale + self.minxx[0]
        Y = Y * self.scale + self.minxx[1]
        Z = Z * self.scale + self.minxx[2]
        
        return potential, X, Y, Z

def cfpurecon(x, nrml, y, gridsize, kernelinfo=None, reginfo=None, n_jobs=None):
    solver = CFPUSolver(kernel_info=kernelinfo, reg_info=reginfo, n_jobs=n_jobs)
    return solver.reconstruct(x, nrml, y, gridsize)

__version__ = '1.3.0-gpu-precision-fix'
