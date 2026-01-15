import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import qr, svd, cholesky
from scipy.optimize import fminbound
from scipy.sparse import coo_matrix
from concurrent.futures import ThreadPoolExecutor
import os
import multiprocessing
from multiprocessing import shared_memory

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

# Global variables for SharedMemory in multiprocessing
_GLOBAL_X = None
_GLOBAL_NRML = None
_SHM_X = None
_SHM_NRML = None


class PolynomialBasis:
    """Handles the computation of curl-free polynomial basis functions."""
    
    @staticmethod
    def compute(x, order):
        """
        Compute the curl-free polynomial basis and the polynomial basis.
        
        Args:
            x (np.ndarray): Input points (N, 3).
            order (int): Polynomial order (1 or 2).
            
        Returns:
            tuple: (CFP, P) where CFP is the curl-free basis gradient and P is the polynomial basis.
        """
        n = x.shape[0]
        if order == 1:
            CP = np.zeros((3 * n, 3))
            P = np.zeros((n, 3))
            P[:, 0:3] = x
            CP[:, 0] = np.tile(np.array([1, 0, 0]), n)
            CP[:, 1] = np.tile(np.array([0, 1, 0]), n)
            CP[:, 2] = np.tile(np.array([0, 0, 1]), n)
            return CP, P
        elif order == 2:
            CP = np.zeros((3 * n, 9))
            P = np.zeros((n, 9))
            P[:, 0:3] = x
            CP[:, 0] = np.tile(np.array([1, 0, 0]), n)
            CP[:, 1] = np.tile(np.array([0, 1, 0]), n)
            CP[:, 2] = np.tile(np.array([0, 0, 1]), n)
            P[:, 3:6] = 0.5 * x**2
            CP[:, 3] = np.concatenate([x[:, 0], np.zeros(n), np.zeros(n)])
            CP[:, 4] = np.concatenate([np.zeros(n), x[:, 1], np.zeros(n)])
            CP[:, 5] = np.concatenate([np.zeros(n), np.zeros(n), x[:, 2]])
            P[:, 6] = x[:, 1] * x[:, 2]
            CP[:, 6] = np.concatenate([np.zeros(n), x[:, 2], x[:, 1]])
            P[:, 7] = x[:, 0] * x[:, 2]
            CP[:, 7] = np.concatenate([x[:, 2], np.zeros(n), x[:, 0]])
            P[:, 8] = x[:, 0] * x[:, 1]
            CP[:, 8] = np.concatenate([x[:, 1], x[:, 0], np.zeros(n)])
            return CP, P
        else:
            raise ValueError('Curl-free polynomial degree not supported')


class WeightFunction:
    """Handles the Partition of Unity weight function."""

    @staticmethod
    def evaluate(r, delta, k=0):
        """
        Evaluate the weight function.
        
        Args:
            r (np.ndarray): Distances.
            delta (float): Support radius.
            k (int): Derivative order (0 or 1).
            
        Returns:
            np.ndarray: Weight values.
        """
        r = r / delta
        phi = np.zeros_like(r)
        if k == 0:
            id1 = r <= (1/3)
            phi[id1] = 0.75 - 2.25 * r[id1]**2
            id2 = (r > 1/3) & (r <= 1)
            phi[id2] = 1.125 * (1 - r[id2])**2
            return phi
        elif k == 1:
            id1 = r <= (1/3)
            phi[id1] = -4.5 / delta**2
            id2 = (r > 1/3) & (r <= 1)
            phi[id2] = (-2.25 * (1 - r[id2]) / delta**2) * (1.0 / r[id2])
            return phi
        else:
            raise ValueError('PU Weight function error: derivative order not supported')


class GCVCost:
    """Handles the Generalized Cross-Validation cost function for regularization."""

    @staticmethod
    def evaluate(lam, z, d, n):
        """
        Calculate the GCV score.
        
        Args:
            lam (float): Log-transformed regularization parameter.
            z (np.ndarray): Transformed data vector.
            d (np.ndarray): Singular values.
            n (float): Effective degrees of freedom factor.
            
        Returns:
            float: GCV score.
        """
        lam = np.exp(-lam)
        temp = (n * lam) / (d**2 + n * lam)
        score = n * np.sum((temp * z)**2) / (np.sum(temp)**2)
        return score


class LocalSolver:
    """Encapsulates the logic for solving the local reconstruction problem on a single patch."""

    @staticmethod
    def solve(x_local, nrml_local, patch_center, patch_radius, h_max, grid_info, kernel_info, reg_info, trbl_local=None):
        """
        Solve the local reconstruction problem.

        Args:
            x_local (np.ndarray): Points in the patch (N, 3).
            nrml_local (np.ndarray): Normals in the patch (N, 3).
            patch_center (tuple): (y0, y1, y2) center of the patch.
            patch_radius (float): Radius of the patch.
            h_max (float): Maximum nearest neighbor distance (for scaling).
            grid_info (dict): Grid parameters (startx, starty, startz, griddx, mmx, mmy, mmz).
            kernel_info (dict): Kernel functions and order.
            reg_info (dict): Regularization parameters.
            trbl_local (np.ndarray, optional): Boolean array indicating troubled points.

        Returns:
            tuple: (idxe_k, Psi_k, potential_k)
                idxe_k: Indices of grid points in the patch.
                Psi_k: PU weights for grid points.
                potential_k: Reconstructed potential values at grid points.
        """
        n = x_local.shape[0]
        if n == 0:
            return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)

        y0, y1, y2 = patch_center
        h2 = h_max**2 if h_max > 0 else 1.0
        
        # Unpack info
        order = kernel_info.get('order', 1)
        eta = kernel_info.get('eta', lambda r: -r)
        zeta = kernel_info.get('zeta', lambda r: -1.0/np.where(r==0, np.inf, r))
        phi = kernel_info.get('phi', lambda r: -r)
        
        exactinterp = reg_info.get('exactinterp', 1)
        nrmlreg = reg_info.get('nrmlreg', 0)
        nrmllambda = reg_info.get('nrmllambda', 0)
        nrmlschur = reg_info.get('nrmlschur', 0)
        potreg = reg_info.get('potreg', 0)
        potlambda = reg_info.get('potlambda', 0)

        startx, starty, startz = grid_info['startx'], grid_info['starty'], grid_info['startz']
        griddx = grid_info['griddx']
        mmx, mmy, mmz = grid_info['mmx'], grid_info['mmy'], grid_info['mmz']

        # Local Coordinates
        xx_local = x_local[:, 0]
        xy_local = x_local[:, 1]
        xz_local = x_local[:, 2]

        # Basis functions
        CFP, P = PolynomialBasis.compute(x_local, order)
        CFPt = CFP.T

        # Pairwise differences
        dx = xx_local.reshape(-1, 1) - xx_local.reshape(1, -1)
        dy = xy_local.reshape(-1, 1) - xy_local.reshape(1, -1)
        dz = xz_local.reshape(-1, 1) - xz_local.reshape(1, -1)
        r = np.sqrt(dx**2 + dy**2 + dz**2)

        # Build System Matrix A
        l_local = 3 if order == 1 else 9
        b = np.zeros(3*n + l_local)
        b[0:3*n:3] = nrml_local[:, 0]
        b[1:3*n:3] = nrml_local[:, 1]
        b[2:3*n:3] = nrml_local[:, 2]

        A = np.zeros((3*n + l_local, 3*n + l_local))
        eta_temp = eta(r)
        zeta_temp = zeta(r)
        np.fill_diagonal(zeta_temp, 0.0)

        dphi_xx = zeta_temp * dx**2 + eta_temp
        dphi_yy = zeta_temp * dy**2 + eta_temp
        dphi_zz = zeta_temp * dz**2 + eta_temp
        dphi_xy = zeta_temp * dx * dy
        dphi_xz = zeta_temp * dx * dz
        dphi_yz = zeta_temp * dy * dz

        A[0:3*n:3, 0:3*n:3] = dphi_xx
        A[0:3*n:3, 1:3*n:3] = dphi_xy
        A[0:3*n:3, 2:3*n:3] = dphi_xz
        A[1:3*n:3, 0:3*n:3] = dphi_xy
        A[1:3*n:3, 1:3*n:3] = dphi_yy
        A[1:3*n:3, 2:3*n:3] = dphi_yz
        A[2:3*n:3, 0:3*n:3] = dphi_xz
        A[2:3*n:3, 1:3*n:3] = dphi_yz
        A[2:3*n:3, 2:3*n:3] = dphi_zz

        A[0:3*n, 3*n:] = CFP
        A[3*n:, 0:3*n] = CFPt

        # Solve for coefficients (Normal constraints)
        coeffs = None
        coeffsp = None

        if nrmlreg != 2:
            if nrmlreg == 1:
                A[0:3*n, 0:3*n] += 3*n*nrmllambda*np.eye(3*n)
            elif nrmlreg == 3:
                if trbl_local is not None and np.any(trbl_local):
                    A[0:3*n, 0:3*n] += 3*n*nrmllambda*np.eye(3*n)
            
            if nrmlschur == 0:
                coeffs_all = np.linalg.solve(A, b)
                coeffsp = coeffs_all[3*n:]
                coeffs = coeffs_all[:3*n]
            else:
                A0 = A[0:3*n, 0:3*n]
                b0 = b[0:3*n]
                AinvCFP = np.linalg.solve(A0, CFP)
                coeffsp = np.linalg.pinv(CFPt @ AinvCFP) @ (CFPt @ np.linalg.solve(A0, b0))
                coeffs = np.linalg.solve(A0, b0 - CFP @ coeffsp)
        else:
            A0 = A[0:3*n, 0:3*n]
            b0 = b[0:3*n]
            Lc = CFP.shape[1]
            F1, G = qr(CFP, mode='economic')
            F2 = F1[:, Lc:]
            F1 = F1[:, :Lc]
            G1 = G[:Lc, :Lc]
            w1 = F1.T @ b0
            w2 = F2.T @ b0
            L = cholesky(F2.T @ A0 @ F2)
            U, D, _ = svd(L.T)
            D = np.diag(D)
            z = U.T @ w2
            lam = fminbound(lambda t: GCVCost.evaluate(t, z, D, 3.0/h2), -10, 35)
            lam = 3.0/h2 * np.exp(-lam)
            A0 = A0 + lam*np.eye(3*n)
            coeffs = F2 @ (U @ (z / (D**2 + lam)))
            coeffsp = np.linalg.solve(G1, w1 - F1.T @ (A0 @ coeffs))

        coeffsx = coeffs[0:3*n:3]
        coeffsy = coeffs[1:3*n:3]
        coeffsz = coeffs[2:3*n:3]
        
        temp_potential_nodes = np.sum(eta_temp * (dx * coeffsx.reshape(1, -1) + 
                                                  dy * coeffsy.reshape(1, -1) + 
                                                  dz * coeffsz.reshape(1, -1)), axis=1) + P @ coeffsp

        # Solve for potential correction (Point constraints)
        coeffs_correction = None
        if exactinterp:
            P0 = np.ones((n, 1))
            A1 = np.ones((n+1, n+1))
            A1[0:n, 0:n] = phi(r)
            A1[-1, -1] = 0.0
            b1 = np.concatenate([temp_potential_nodes, np.array([0.0])])

            # DEBUG: Print b1 stats
            if not hasattr(LocalSolver, "debug_counter_b1"):
                 LocalSolver.debug_counter_b1 = 0
            if LocalSolver.debug_counter_b1 < 3:
                 print(f"CPU b1 Patch {LocalSolver.debug_counter_b1}: Mean={np.mean(b1[:-1]):.6e}, Max={np.max(b1[:-1]):.6e}, AbsMax={np.max(np.abs(b1[:-1])):.6e}")
                 LocalSolver.debug_counter_b1 += 1

            if potreg != 2:
                if potreg == 1:
                    A1[0:n, 0:n] += n*potlambda*np.eye(n)
                elif potreg == 3:
                    if trbl_local is not None and np.any(trbl_local):
                        A1[0:n, 0:n] += n*potlambda*np.eye(n)
                coeffs_correction = np.linalg.solve(A1, b1)
            else:
                Lc = P0.shape[1]
                b2 = b1[0:n]
                A2 = A1[0:n, 0:n]
                F1, G = qr(P0, mode='economic')
                F2 = F1[:, Lc:]
                F1 = F1[:, :Lc]
                G1 = G[:Lc, :Lc]
                w1 = F1.T @ b2
                w2 = F2.T @ b2
                L = cholesky(F2.T @ A2 @ F2)
                U, D, _ = svd(L.T)
                D = np.diag(D)
                z2 = U.T @ w2
                lam = fminbound(lambda t: GCVCost.evaluate(t, z2, D, 1.0/h2), -10, 35)
                lam = (1.0/h2) * np.exp(-lam)
                A2 = A2 + lam*np.eye(n)
                temp = F2 @ (U @ (z2 / (D**2 + lam)))
                coeffs_correction = np.concatenate([temp, np.linalg.solve(G1, w1 - F1.T @ (A2 @ temp))])
        else:
            P1 = np.hstack([P[:, 0:3], np.ones((n, 1))])
            coeffs_correction = np.linalg.lstsq(P1, temp_potential_nodes, rcond=None)[0]

        coeffs_correction_const = coeffs_correction[-1]
        coeffs_correction_vec = coeffs_correction[:-1]

        # DEBUG: Print Coeffs Stats (Moved to end)
        if not hasattr(LocalSolver, "debug_counter"):
             LocalSolver.debug_counter = 0
        if LocalSolver.debug_counter < 3:
             print(f"CPU Coeffs Patch {LocalSolver.debug_counter}: Sum={np.sum(coeffs):.6e}, AbsSum={np.sum(np.abs(coeffs)):.6e}")
             if coeffs_correction is not None:
                 print(f"CPU Corr Coeffs Patch {LocalSolver.debug_counter}: Sum={np.sum(coeffs_correction):.6e}, AbsSum={np.sum(np.abs(coeffs_correction)):.6e}")
             LocalSolver.debug_counter += 1

        # Evaluation on grid points
        ix = int(np.round((y0 - startx) / griddx)) + 1
        iy = int(np.round((y1 - starty) / griddx)) + 1
        iz = int(np.round((y2 - startz) / griddx)) + 1
        factor = int(np.round(patch_radius / griddx))
        
        ixs = np.arange(max(ix - factor, 1), min(ix + factor, mmx) + 1)
        iys = np.arange(max(iy - factor, 1), min(iy + factor, mmy) + 1)
        izs = np.arange(max(iz - factor, 1), min(iz + factor, mmz) + 1)
        
        xxg = startx + (ixs - 1) * griddx
        yyg = starty + (iys - 1) * griddx
        zzg = startz + (izs - 1) * griddx
        
        XX3, YY3, ZZ3 = np.meshgrid(xxg, yyg, zzg, indexing='xy')
        De = (y0 - XX3)**2 + (y1 - YY3)**2 + (y2 - ZZ3)**2
        idmask = De.reshape(-1) < patch_radius**2
        
        # Grid indices mapping
        ixs2 = np.repeat(ixs.reshape(1, -1), len(yyg), axis=0)
        ixs2 = np.repeat(ixs2[:, :, np.newaxis], len(zzg), axis=2)
        iys2 = np.repeat(iys.reshape(-1, 1), len(xxg), axis=1)
        iys2 = np.repeat(iys2[:, :, np.newaxis], len(zzg), axis=2)
        izs2 = np.repeat(izs.reshape(1, 1, -1), len(yyg), axis=0)
        izs2 = np.repeat(izs2, len(xxg), axis=1)
        
        temp_idg = (iys2 + (ixs2 - 1) * mmy) + (izs2 - 1) * (mmx * mmy)
        temp_idg = temp_idg.reshape(-1)
        temp_idg = temp_idg[idmask] - 1
        
        De = np.sqrt(De.reshape(-1)[idmask])
        idxe_k = temp_idg.astype(int)
        Psi_k = WeightFunction.evaluate(De, patch_radius, 0)
        
        xe_local = np.vstack([XX3.reshape(-1), YY3.reshape(-1), ZZ3.reshape(-1)]).T
        xe_local = xe_local[idmask, :]
        mm = xe_local.shape[0]
        
        if mm == 0:
            return idxe_k, Psi_k, np.array([], dtype=float)

        # Batch evaluation
        batch_sz = int(np.ceil(100**2 / max(n, 1)))
        temp_potential = np.zeros(mm)
        potential_correction = np.zeros(mm)
        
        for j in range(0, mm, batch_sz):
            idb = slice(j, min(j + batch_sz, mm))
            xe_local_batch = xe_local[idb, :]
            
            dxb = xe_local_batch[:, 0].reshape(-1, 1) - xx_local.reshape(1, -1)
            dyb = xe_local_batch[:, 1].reshape(-1, 1) - xy_local.reshape(1, -1)
            dzb = xe_local_batch[:, 2].reshape(-1, 1) - xz_local.reshape(1, -1)
            rb = np.sqrt(dxb**2 + dyb**2 + dzb**2)
            
            _, Pb = PolynomialBasis.compute(xe_local_batch, order)
            
            temp_potential[j:j+xe_local_batch.shape[0]] = np.sum(eta(rb) * (
                dxb * coeffsx.reshape(1, -1) + 
                dyb * coeffsy.reshape(1, -1) + 
                dzb * coeffsz.reshape(1, -1)), axis=1) + Pb @ coeffsp
            
            if exactinterp:
                potential_correction[j:j+xe_local_batch.shape[0]] = phi(rb) @ coeffs_correction_vec + coeffs_correction_const
            else:
                potential_correction[j:j+xe_local_batch.shape[0]] = Pb[:, 0:3] @ coeffs_correction_vec + coeffs_correction_const
                
        potential_k = temp_potential - potential_correction
        return idxe_k, Psi_k, potential_k


# --- Multiprocessing Helper Functions ---

def _init_proc(shm_name_x, shape_x, dtype_x, shm_name_nrml, shape_nrml, dtype_nrml):
    """Initializer for worker processes to attach to shared memory."""
    if shared_memory is None:
        return
    global _GLOBAL_X, _GLOBAL_NRML, _SHM_X, _SHM_NRML
    _SHM_X = shared_memory.SharedMemory(name=shm_name_x)
    _SHM_NRML = shared_memory.SharedMemory(name=shm_name_nrml)
    _GLOBAL_X = np.ndarray(shape_x, dtype=np.dtype(dtype_x), buffer=_SHM_X.buf)
    _GLOBAL_NRML = np.ndarray(shape_nrml, dtype=np.dtype(dtype_nrml), buffer=_SHM_NRML.buf)


def _compute_proc(args):
    """Worker function for multiprocessing."""
    (k, idk, nn_dist_k, y0, y1, y2, patchRad_k, kernel_info, reg_info, 
     grid_info, trbl_local) = args
     
    if idk.size == 0:
        return (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float))

    x_local = _GLOBAL_X[idk, :]
    nrml_local = _GLOBAL_NRML[idk, :]
    
    h_max = np.max(nn_dist_k) if nn_dist_k.size else 1.0
    patch_center = (y0, y1, y2)
    
    idxe_k, Psi_k, potential_k = LocalSolver.solve(
        x_local, nrml_local, patch_center, patchRad_k, h_max, 
        grid_info, kernel_info, reg_info, trbl_local
    )
    
    patch_vec_k = np.full(idxe_k.shape[0], k + 1)
    return (idxe_k, patch_vec_k, Psi_k, potential_k)


# --- Default Kernel Functions (Picklable) ---

def _default_phi(r):
    return -r

def _default_eta(r):
    return -r

def _default_zeta(r):
    return -1.0/np.where(r==0, np.inf, r)


class CFPUSolver:
    """
    Main class for Curl-Free Partition of Unity Reconstruction.
    Refactors the functionality of cfpurecon.
    """

    def __init__(self, kernel_info=None, reg_info=None, n_jobs=None):
        if kernel_info is None:
            self.kernel_info = {
                'phi': _default_phi,
                'eta': _default_eta,
                'zeta': _default_zeta,
                'order': 1
            }
        else:
            self.kernel_info = kernel_info
            
        self.reg_info = reg_info if reg_info is not None else {'exactinterp': 1}
        self.n_jobs = n_jobs
        
    def fit(self, x, nrml):
        """Pre-process data (normalization)."""
        self.minxx = np.min(x, axis=0)
        self.maxxx = np.max(x, axis=0)
        scale = np.max(self.maxxx - self.minxx)
        
        self.x_norm = (x - self.minxx) / scale
        self.nrml = nrml # Normals don't need scaling usually, but orientation matters
        self.scale = scale
        return self

    def reconstruct(self, x, nrml, centers, gridsize):
        """
        Run the reconstruction.
        
        Args:
            x (np.ndarray): Input points (N, 3).
            nrml (np.ndarray): Input normals (N, 3).
            centers (np.ndarray): Patch centers (M, 3).
            gridsize (tuple or float): Grid resolution/size.
            
        Returns:
            tuple: (potential, X, Y, Z)
        """
        # Normalization
        self.fit(x, nrml)
        y_norm = (centers - self.minxx) / self.scale
        
        M = y_norm.shape[0]
        N = self.x_norm.shape[0]
        
        # Build Trees
        tree_y = cKDTree(y_norm)
        nn_dist = tree_y.query(y_norm, k=2)[0][:, 1]
        H = np.max(nn_dist)
        delta = 1.0
        patchRad0 = (1.0 + delta) * H / 2.0
        
        tree_x = cKDTree(self.x_norm)
        idx = []
        nn_dist_list = []
        
        # Patch assignment
        for k in range(M):
            id_list = tree_x.query_ball_point(y_norm[k, :], patchRad0)
            idx.append(np.array(id_list, dtype=int))
            if len(id_list) == 0:
                nn_dist_list.append(np.array([], dtype=float))
            else:
                dists = np.linalg.norm(self.x_norm[id_list, :] - y_norm[k, :], axis=1)
                nn_dist_list.append(dists)
                
        patchRad = np.full(M, patchRad0)
        nodeInPatch = np.zeros(N, dtype=bool)
        for k in range(M):
            nodeInPatch[idx[k]] = True
            
        # Ensure coverage
        missingIds = np.where(~nodeInPatch)[0]
        while missingIds.size > 0:
            cp_id = tree_y.query(self.x_norm[missingIds[0], :], k=1)[1]
            p_dist = tree_y.query(self.x_norm[missingIds[0], :], k=1)[0]
            temp_rad = 1.01 * p_dist
            id_list = tree_x.query_ball_point(y_norm[cp_id, :], temp_rad)
            dists = np.linalg.norm(self.x_norm[id_list, :] - y_norm[cp_id, :], axis=1)
            idx[cp_id] = np.array(id_list, dtype=int)
            nn_dist_list[cp_id] = dists
            patchRad[cp_id] = temp_rad
            nodeInPatch[id_list] = True
            missingIds = np.where(~nodeInPatch)[0]

        # Grid setup
        minx = np.min(self.x_norm, axis=0)
        maxx = np.max(self.x_norm, axis=0)
        griddx = np.max((maxx - minx) / gridsize)
        
        startx = minx[0] - 3 * griddx
        endx = maxx[0] + 3 * griddx
        starty = minx[1] - 3 * griddx
        endy = maxx[1] + 3 * griddx
        startz = minx[2] - 3 * griddx
        endz = maxx[2] + 3 * griddx
        
        xx = np.arange(startx, endx + griddx/2, griddx)
        yy = np.arange(starty, endy + griddx/2, griddx)
        zz = np.arange(startz, endz + griddx/2, griddx)
        
        X, Y, Z = np.meshgrid(xx, yy, zz, indexing='xy')
        mmy, mmx, mmz = X.shape
        m = mmx * mmy * mmz
        
        grid_info = {
            'startx': startx, 'starty': starty, 'startz': startz,
            'griddx': griddx,
            'mmx': mmx, 'mmy': mmy, 'mmz': mmz
        }
        
        trbl_id = self.reg_info.get('trbl_id', np.zeros(N, dtype=bool))
        
        # Parallel Execution
        idxe_patch = [None] * M
        patch_vec = [None] * M
        Psi = [None] * M
        potential_local = [None] * M
        
        workers = self.n_jobs if (self.n_jobs and self.n_jobs > 0) else min(M, os.cpu_count() or 1)
        mode_env = os.environ.get('CFPU_PARALLEL', 'thread')
        
        use_multiprocessing = (workers > 1 and mode_env == 'process' and 
                               shared_memory is not None and multiprocessing.get_context is not None)
        
        if use_multiprocessing:
            shm_x = shared_memory.SharedMemory(create=True, size=self.x_norm.nbytes)
            shm_nrml = shared_memory.SharedMemory(create=True, size=self.nrml.nbytes)
            np.ndarray(self.x_norm.shape, dtype=self.x_norm.dtype, buffer=shm_x.buf)[:] = self.x_norm
            np.ndarray(self.nrml.shape, dtype=self.nrml.dtype, buffer=shm_nrml.buf)[:] = self.nrml
            
            try:
                ctx = multiprocessing.get_context('spawn')
                # Reset globals in main process just in case, though they are for workers
                
                # Note: kernel_info contains lambdas which cannot be pickled easily.
                # If using multiprocessing, we must ensure everything passed is picklable.
                # Lambdas in self.kernel_info might be an issue.
                # For now, assuming they are picklable or the default ones are used which might work if defined at module level.
                # Actually, lambdas are NOT picklable. We need to handle this.
                # Since the default kernel_info uses lambdas, we need to convert them to named functions or handle them inside the worker.
                # To support custom kernels in MP, users should pass picklable functions.
                # For default, we can reconstruct them in worker if they are missing or standard.
                
                # Workaround: if kernel_info has lambdas, we might fail. 
                # Ideally, we should define the default functions at module level.
                
                arg_iter = ((k, idx[k], nn_dist_list[k], y_norm[k, 0], y_norm[k, 1], y_norm[k, 2], 
                             patchRad[k], self.kernel_info, self.reg_info, grid_info, trbl_id[idx[k]]) 
                            for k in range(M))
                            
                with ctx.Pool(processes=workers, initializer=_init_proc, 
                              initargs=(shm_x.name, self.x_norm.shape, self.x_norm.dtype.str, 
                                        shm_nrml.name, self.nrml.shape, self.nrml.dtype.str)) as pool:
                    for k, res in enumerate(pool.imap(_compute_proc, arg_iter)):
                        idxe_patch[k], patch_vec[k], Psi[k], potential_local[k] = res
            finally:
                shm_x.close()
                shm_x.unlink()
                shm_nrml.close()
                shm_nrml.unlink()
                
        else:
            # Threading or Serial
            def _compute_thread(k):
                idk = idx[k]
                if idk.size == 0:
                    return (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float))
                
                x_local = self.x_norm[idk, :]
                nrml_local = self.nrml[idk, :]
                h_max = np.max(nn_dist_list[k]) if nn_dist_list[k].size else 1.0
                patch_center = (y_norm[k, 0], y_norm[k, 1], y_norm[k, 2])
                
                idxe_k, Psi_k, potential_k = LocalSolver.solve(
                    x_local, nrml_local, patch_center, patchRad[k], h_max, 
                    grid_info, self.kernel_info, self.reg_info, trbl_id[idk]
                )
                patch_vec_k = np.full(idxe_k.shape[0], k + 1)
                return (idxe_k, patch_vec_k, Psi_k, potential_k)

            if workers > 1:
                # Threading
                if threadpool_limits is not None:
                    blas_threads_env = os.environ.get('CFPU_BLAS_THREADS')
                    blas_threads = int(blas_threads_env) if blas_threads_env else 1
                    with threadpool_limits(limits=blas_threads, user_api='blas'):
                        with ThreadPoolExecutor(max_workers=workers) as ex:
                            for k, res in enumerate(ex.map(_compute_thread, range(M))):
                                idxe_patch[k], patch_vec[k], Psi[k], potential_local[k] = res
                else:
                    with ThreadPoolExecutor(max_workers=workers) as ex:
                        for k, res in enumerate(ex.map(_compute_thread, range(M))):
                            idxe_patch[k], patch_vec[k], Psi[k], potential_local[k] = res
            else:
                # Serial
                for k in range(M):
                    idxe_patch[k], patch_vec[k], Psi[k], potential_local[k] = _compute_thread(k)

        # Blending Results (Partition of Unity)
        patch_vec_cat = np.concatenate([pv for pv in patch_vec if pv is not None and pv.size > 0]) if any([pv is not None and pv.size > 0 for pv in patch_vec]) else np.array([], dtype=int)
        idxe_vec_cat = np.concatenate([ie for ie in idxe_patch if ie is not None and ie.size > 0]) if any([ie is not None and ie.size > 0 for ie in idxe_patch]) else np.array([], dtype=int)
        Psi_cat = np.concatenate([ps for ps in Psi if ps is not None and ps.size > 0]) if any([ps is not None and ps.size > 0 for ps in Psi]) else np.array([], dtype=float)
        
        Psi_sum = np.zeros(m)
        if idxe_vec_cat.size > 0:
            Psi_sum = coo_matrix((Psi_cat, (idxe_vec_cat, patch_vec_cat - 1)), shape=(m, M)).sum(axis=1).A1
            
        for k in range(M):
            if potential_local[k] is not None and potential_local[k].size > 0:
                denom = Psi_sum[idxe_patch[k]]
                potential_local[k] = potential_local[k] * (Psi[k] / denom)
                
        temp = np.zeros(m)
        potential_local_cat = np.concatenate([pl for pl in potential_local if pl is not None and pl.size > 0]) if any([pl is not None and pl.size > 0 for pl in potential_local]) else np.array([], dtype=float)
        
        if idxe_vec_cat.size > 0:
            temp = coo_matrix((potential_local_cat, (idxe_vec_cat, patch_vec_cat - 1)), shape=(m, M)).sum(axis=1).A1
            
        # DEBUG: Print Stats
        print(f"CPU Psi_sum: Mean={np.mean(Psi_sum):.6e}, Max={np.max(Psi_sum):.6e}, Sum={np.sum(Psi_sum):.6e}")
        print(f"CPU Potential Sum (Pre-Div): Mean={np.mean(temp):.6e}, Max={np.max(temp):.6e}, Sum={np.sum(temp):.6e}")

        i_nonzero = np.where(Psi_sum > 0)[0]
        potential = np.full(m, np.nan)
        potential[i_nonzero] = temp[i_nonzero]
        potential = potential.reshape((mmy, mmx, mmz), order='F')
        
        # Scale grid back
        X = X * self.scale + self.minxx[0]
        Y = Y * self.scale + self.minxx[1]
        Z = Z * self.scale + self.minxx[2]
        
        return potential, X, Y, Z

def cfpurecon(x, nrml, y, gridsize, kernelinfo=None, reginfo=None, n_jobs=None):
    """
    Wrapper function for backward compatibility.
    """
    solver = CFPUSolver(kernel_info=kernelinfo, reg_info=reginfo, n_jobs=n_jobs)
    return solver.reconstruct(x, nrml, y, gridsize)
