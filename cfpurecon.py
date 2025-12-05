import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import qr, svd, cholesky
from scipy.optimize import fminbound
from scipy.sparse import coo_matrix
from .util.curlfree_poly import curlfree_poly
from .util.weight import weight
from .util.gcv_cost_function import gcv_cost_function

def cfpurecon(x, nrml, y, gridsize, kernelinfo=None, reginfo=None):
    if kernelinfo is None:
        kernelinfo = {
            'phi': lambda r: -r,
            'eta': lambda r: -r,
            'zeta': lambda r: -1.0/np.where(r==0, np.inf, r),
            'order': 1
        }
    if reginfo is None:
        reginfo = {'exactinterp': 1}
    minxx = np.min(x, axis=0)
    maxxx = np.max(x, axis=0)
    x = x - minxx
    scale = np.max(maxxx - minxx)
    x = x / scale
    y = y - minxx
    y = y / scale
    M = y.shape[0]
    tree_y = cKDTree(y)
    nn_dist = tree_y.query(y, k=2)[0][:, 1]
    H = np.max(nn_dist)
    delta = 1.0
    patchRad0 = (1.0 + delta) * H / 2.0
    N = x.shape[0]
    tree_x = cKDTree(x)
    idx = []
    nn_dist_list = []
    for k in range(M):
        id_list = tree_x.query_ball_point(y[k, :], patchRad0)
        idx.append(np.array(id_list, dtype=int))
        if len(id_list) == 0:
            nn_dist_list.append(np.array([], dtype=float))
        else:
            dists = np.linalg.norm(x[id_list, :] - y[k, :], axis=1)
            nn_dist_list.append(dists)
    patchRad = np.full(M, patchRad0)
    nodeInPatch = np.zeros(N, dtype=bool)
    for k in range(M):
        nodeInPatch[idx[k]] = True
    missingIds = np.where(~nodeInPatch)[0]
    while missingIds.size > 0:
        cp_id = tree_y.query(x[missingIds[0], :], k=1)[1]
        p_dist = tree_y.query(x[missingIds[0], :], k=1)[0]
        temp_rad = 1.01 * p_dist
        id_list = tree_x.query_ball_point(y[cp_id, :], temp_rad)
        dists = np.linalg.norm(x[id_list, :] - y[cp_id, :], axis=1)
        idx[cp_id] = np.array(id_list, dtype=int)
        nn_dist_list[cp_id] = dists
        patchRad[cp_id] = temp_rad
        nodeInPatch[id_list] = True
        missingIds = np.where(~nodeInPatch)[0]
    exactinterp = reginfo.get('exactinterp', 1)
    nrmlreg = reginfo.get('nrmlreg', 0)
    nrmllambda = reginfo.get('nrmllambda', 0)
    nrmlschur = reginfo.get('nrmlschur', 0)
    trbl_id = reginfo.get('trbl_id', np.zeros(N, dtype=bool))
    potreg = reginfo.get('potreg', 0)
    potlambda = reginfo.get('potlambda', 0)
    eta = kernelinfo['eta']
    zeta = kernelinfo['zeta']
    phi = kernelinfo['phi']
    order = kernelinfo['order']
    if order == 1:
        l = 3
    elif order == 2:
        l = 9
    else:
        raise ValueError('Curl-free polynomial degree not supported')
    minx = np.min(x, axis=0)
    maxx = np.max(x, axis=0)
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
    idxe_patch = [None] * M
    patch_vec = [None] * M
    Psi = [None] * M
    potential_local = [None] * M
    for k in range(M):
        idk = idx[k]
        if idk.size == 0:
            idxe_patch[k] = np.array([], dtype=int)
            patch_vec[k] = np.array([], dtype=int)
            Psi[k] = np.array([], dtype=float)
            potential_local[k] = np.array([], dtype=float)
            continue
        h2 = np.max(nn_dist_list[k])**2 if nn_dist_list[k].size else 1.0
        x_local = x[idk, :]
        xx_local = x_local[:, 0]
        xy_local = x_local[:, 1]
        xz_local = x_local[:, 2]
        n = x_local.shape[0]
        CFP, P = curlfree_poly(x_local, order)
        CFPt = CFP.T
        dx = xx_local.reshape(-1, 1) - xx_local.reshape(1, -1)
        dy = xy_local.reshape(-1, 1) - xy_local.reshape(1, -1)
        dz = xz_local.reshape(-1, 1) - xz_local.reshape(1, -1)
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        ui = nrml[idk, :]
        b = np.zeros(3*n + l)
        b[0:3*n:3] = ui[:, 0]
        b[1:3*n:3] = ui[:, 1]
        b[2:3*n:3] = ui[:, 2]
        A = np.zeros((3*n + l, 3*n + l))
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
        if nrmlreg != 2:
            if nrmlreg == 1:
                A[0:3*n, 0:3*n] = A[0:3*n, 0:3*n] + 3*n*nrmllambda*np.eye(3*n)
            elif nrmlreg == 3:
                if np.any(trbl_id[idk]):
                    A[0:3*n, 0:3*n] = A[0:3*n, 0:3*n] + 3*n*nrmllambda*np.eye(3*n)
            if nrmlschur == 0:
                coeffs = np.linalg.solve(A, b)
                coeffsp = coeffs[3*n:]
                coeffs = coeffs[:3*n]
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
            lam = fminbound(lambda t: gcv_cost_function(t, z, D, 3.0/h2), -10, 35)
            lam = 3.0/h2 * np.exp(-lam)
            A0 = A0 + lam*np.eye(3*n)
            coeffs = F2 @ (U @ (z / (D**2 + lam)))
            coeffsp = np.linalg.solve(G1, w1 - F1.T @ (A0 @ coeffs))
        coeffsx = coeffs[0:3*n:3]
        coeffsy = coeffs[1:3*n:3]
        coeffsz = coeffs[2:3*n:3]
        temp_potential_nodes = np.sum(eta_temp * (dx * coeffsx.reshape(1, -1) + dy * coeffsy.reshape(1, -1) + dz * coeffsz.reshape(1, -1)), axis=1) + P @ coeffsp
        if exactinterp:
            P0 = np.ones((n, 1))
            A1 = np.ones((n+1, n+1))
            A1[0:n, 0:n] = phi(r)
            A1[-1, -1] = 0.0
            b1 = np.concatenate([temp_potential_nodes, np.array([0.0])])
            if potreg != 2:
                if potreg == 1:
                    A1[0:n, 0:n] = A1[0:n, 0:n] + n*potlambda*np.eye(n)
                elif potreg == 3:
                    if np.any(trbl_id[idk]):
                        A1[0:n, 0:n] = A1[0:n, 0:n] + n*potlambda*np.eye(n)
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
                lam = fminbound(lambda t: gcv_cost_function(t, z2, D, 1.0/h2), -10, 35)
                lam = (1.0/h2) * np.exp(-lam)
                A2 = A2 + lam*np.eye(n)
                temp = F2 @ (U @ (z2 / (D**2 + lam)))
                coeffs_correction = np.concatenate([temp, np.linalg.solve(G1, w1 - F1.T @ (A2 @ temp))])
        else:
            P1 = np.hstack([P[:, 0:3], np.ones((n, 1))])
            coeffs_correction = np.linalg.lstsq(P1, temp_potential_nodes, rcond=None)[0]
        coeffs_correction_const = coeffs_correction[-1]
        coeffs_correction_vec = coeffs_correction[:-1]
        ix = int(np.round((y[k, 0] - startx) / griddx)) + 1
        iy = int(np.round((y[k, 1] - starty) / griddx)) + 1
        iz = int(np.round((y[k, 2] - startz) / griddx)) + 1
        factor = int(np.round(patchRad[k] / griddx))
        ixs = np.arange(max(ix - factor, 1), min(ix + factor, mmx) + 1)
        iys = np.arange(max(iy - factor, 1), min(iy + factor, mmy) + 1)
        izs = np.arange(max(iz - factor, 1), min(iz + factor, mmz) + 1)
        xxg = startx + (ixs - 1) * griddx
        yyg = starty + (iys - 1) * griddx
        zzg = startz + (izs - 1) * griddx
        XX3, YY3, ZZ3 = np.meshgrid(xxg, yyg, zzg, indexing='xy')
        De = (y[k, 0] - XX3)**2 + (y[k, 1] - YY3)**2 + (y[k, 2] - ZZ3)**2
        idmask = De.reshape(-1) < patchRad[k]**2
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
        idxe_patch[k] = temp_idg.astype(int)
        Psi[k] = weight(De, patchRad[k], 0)
        mlocalx = len(xxg)
        mlocaly = len(yyg)
        mlocalz = len(zzg)
        xe_local = np.vstack([XX3.reshape(-1), YY3.reshape(-1), ZZ3.reshape(-1)]).T
        xe_local = xe_local[idmask, :]
        mm = xe_local.shape[0]
        if mm == 0:
            potential_local[k] = np.array([], dtype=float)
            patch_vec[k] = np.array([], dtype=int)
            continue
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
            _, Pb = curlfree_poly(xe_local_batch, order)
            temp_potential[j:j+xe_local_batch.shape[0]] = np.sum(eta(rb) * (dxb * coeffsx.reshape(1, -1) + dyb * coeffsy.reshape(1, -1) + dzb * coeffsz.reshape(1, -1)), axis=1) + Pb @ coeffsp
            if exactinterp:
                potential_correction[j:j+xe_local_batch.shape[0]] = phi(rb) @ coeffs_correction_vec + coeffs_correction_const
            else:
                potential_correction[j:j+xe_local_batch.shape[0]] = Pb[:, 0:3] @ coeffs_correction_vec + coeffs_correction_const
        potential_local[k] = temp_potential - potential_correction
        patch_vec[k] = np.full(mm, k + 1)
    patch_vec_cat = np.concatenate([pv for pv in patch_vec if pv.size > 0]) if any([pv.size > 0 for pv in patch_vec]) else np.array([], dtype=int)
    idxe_vec_cat = np.concatenate([ie for ie in idxe_patch if ie.size > 0]) if any([ie.size > 0 for ie in idxe_patch]) else np.array([], dtype=int)
    Psi_cat = np.concatenate([ps for ps in Psi if ps.size > 0]) if any([ps.size > 0 for ps in Psi]) else np.array([], dtype=float)
    Psi_sum = np.zeros(m)
    if idxe_vec_cat.size > 0:
        Psi_sum = coo_matrix((Psi_cat, (idxe_vec_cat, patch_vec_cat - 1)), shape=(m, M)).sum(axis=1).A1
    for k in range(M):
        if potential_local[k].size > 0:
            denom = Psi_sum[idxe_patch[k]]
            potential_local[k] = potential_local[k] * (Psi[k] / denom)
    temp = np.zeros(m)
    if idxe_vec_cat.size > 0:
        temp = coo_matrix((np.concatenate([pl for pl in potential_local if pl.size > 0]), (idxe_vec_cat, patch_vec_cat - 1)), shape=(m, M)).sum(axis=1).A1
    i_nonzero = np.where(Psi_sum > 0)[0]
    potential = np.full(m, np.nan)
    potential[i_nonzero] = temp[i_nonzero]
    potential = potential.reshape((mmy, mmx, mmz), order='F')
    X = X * scale + minxx[0]
    Y = Y * scale + minxx[1]
    Z = Z * scale + minxx[2]
    return potential, X, Y, Z
