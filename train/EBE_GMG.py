import torch.nn as nn
import time
import numpy as np
import numpy as np
import torch
from omegaconf import ListConfig

import spconv.pytorch as spconv
from types import SimpleNamespace

class ProlongStencil(nn.Module):
    """Stateless matrix-free prolongation / restriction.

    No internal stencil state (no self.c_idx / self.w), to avoid device mismatch.
    You must pass (c_idx, w) explicitly each call.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def _align_stencil(c_idx: torch.Tensor, w: torch.Tensor, device, w_dtype):
        # c_idx must be long on same device; w must be float on same device/dtype
        if c_idx.device != device:
            c_idx = c_idx.to(device=device)
        if c_idx.dtype != torch.long:
            c_idx = c_idx.to(dtype=torch.long)
        if w.device != device or w.dtype != w_dtype:
            w = w.to(device=device, dtype=w_dtype)
        return c_idx, w

    def prolong(self, u_coarse: torch.Tensor, c_idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """u_f = P * u_c
        u_coarse: (nC,3), c_idx/w: (nF,8)
        """
        nC = u_coarse.shape[0]
        nF = c_idx.shape[0]
        if nF == 0:
            return u_coarse.new_zeros((0, 3))
        if nC == 0:
            return u_coarse.new_zeros((nF, 3))

        c_idx, w = self._align_stencil(c_idx, w, u_coarse.device, u_coarse.dtype)

        ok = (c_idx >= 0) & (c_idx < nC)             
        idx_safe = c_idx.clamp(0, nC - 1)

        uc = u_coarse[idx_safe]                       # (nF,8,3)
        contrib = uc * w.unsqueeze(-1)                # (nF,8,3)
        contrib = contrib * ok.unsqueeze(-1)          # invalid -> 0
        return contrib.sum(dim=1)                     # (nF,3)

    def restrict_PT(self, r_fine: torch.Tensor, nC: int, c_idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """r_c = P^T * r_f
        r_fine: (nF,3), return: (nC,3)
        """
        if nC == 0:
            return r_fine.new_zeros((0, 3))
        nF = c_idx.shape[0]
        if nF == 0:
            return r_fine.new_zeros((nC, 3))

        c_idx, w = self._align_stencil(c_idx, w, r_fine.device, r_fine.dtype)

        ok = (c_idx >= 0) & (c_idx < nC)
        idx_safe = c_idx.clamp(0, nC - 1)

        contrib = r_fine.unsqueeze(1) * w.unsqueeze(-1)   # (nF,8,3)
        contrib = contrib * ok.unsqueeze(-1)

        rc = torch.zeros((nC, 3), device=r_fine.device, dtype=r_fine.dtype)
        rc.index_add_(0, idx_safe.reshape(-1), contrib.reshape(-1, 3))
        return rc

def assembly_F(voxel, Fe):
        """assemble stiffness matrix K and force matrix for solid cell

        Args:
           voxel (torch.cuda.FloatTensor): Voxel, it should be N*N*N 
           Ke (torch.cuda.FloatTensor): Elelment stiffness matrix, it should be 24*24 
           Fe (torch.cuda.FloatTensor): Elelment macrostrain-force matrix, it should be 24*6 

        Returns:
            K(torch.cuda.Soarse_COO_Tensor)
            F(torch.cuda.FloatTensor)
        """
        device = voxel.device
        voxel_numel=torch.count_nonzero(voxel)
        voxel_coo=torch.nonzero(voxel, as_tuple=False)
        node=voxel.clone().long()
        hex=torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=torch.int,device=device)
        for i in hex:
            tmp=(voxel_coo+i+voxel.shape[0])%voxel.shape[0]
            node[tmp[:, 0], tmp[:, 1], tmp[:, 2]] = 1
        node_numel=torch.count_nonzero(node)
        node_coo=torch.nonzero(node, as_tuple=False)
        nodeidx = torch.arange(0, node_numel,device=device)
        # print(node_coo.shape,node.shape,torch.max(node_coo))
        node[node_coo[:, 0], node_coo[:, 1], node_coo[:, 2]]=nodeidx

        dof = torch.zeros(voxel_numel,8, device=device)
        for i in range(0,8):
            now_node_coo=(voxel_coo+hex[i]+voxel.shape[0])%voxel.shape[0]
            dof[:,i] = node[now_node_coo[:,0],now_node_coo[:,1],now_node_coo[:,2]]

        fij = torch.zeros(2,voxel_numel * 8 * 3 * 6, dtype=torch.float, device=device)
        fij[0,:]=dof.reshape(-1,1).repeat(1, 3 * 6).reshape(-1)
        fij[1,:]=torch.arange(0, 3 * 6, device=device).reshape(1,1,-1).repeat(voxel_numel, 8, 1).contiguous().reshape(-1)
        vF = Fe.repeat(voxel_numel, 1, 1).reshape(-1)
        
        F = torch.sparse_coo_tensor(fij, vF.contiguous().reshape(-1), (node_numel,3*6), device=device).coalesce().to_dense()

        return F.reshape(-1,3,6)[:,:,0]

# ==========================================
# ==========================================
def isotropic_elastic_tensor(E, v):

    Lambda = v / (1. + v) / (1 - 2. * v) * E
    Mu = 1. / (2. * (1. + v)) * E
    return torch.as_tensor([
        [Lambda + 2 * Mu, Lambda, Lambda, 0, 0, 0],
        [Lambda, Lambda + 2 * Mu, Lambda, 0, 0, 0],
        [Lambda, Lambda, Lambda + 2 * Mu, 0, 0, 0],
        [0, 0, 0, Mu, 0, 0],
        [0, 0, 0, 0, Mu, 0],
        [0, 0, 0, 0, 0, Mu]])

def hexahedron(C, r):

    device = C.device
    dx = 1 / r / 2.
    dy = 1 / r / 2.
    dz = 1 / r / 2.

    pp = torch.as_tensor([-pow(3 / 5, 0.5), 0, pow(3 / 5, 0.5)], dtype=C.dtype, device=device)
    ww = torch.as_tensor([5 / 9, 8 / 9, 5 / 9], dtype=C.dtype, device=device)
    Ke = torch.zeros(24, 24, dtype=C.dtype, device=device)
    Fe = torch.zeros(24, 6, dtype=C.dtype, device=device)

    dxdydz = torch.as_tensor(
        [[-dx, dx, dx, -dx, -dx, dx, dx, -dx], [-dy, -dy, dy, dy, -dy, -dy, dy, dy],
         [-dz, -dz, -dz, -dz, dz, dz, dz, dz]], dtype=C.dtype, device=device).t()

    for i in range(3):
        for j in range(3):
            for k in range(3):
                x, y, z = pp[i], pp[j], pp[k]
                qxqyqz = torch.as_tensor(
                    [[-((y - 1) * (z - 1)) / 8, ((y - 1) * (z - 1)) / 8, -((y + 1) * (z - 1)) / 8,
                      ((y + 1) * (z - 1)) / 8, ((y - 1) * (z + 1)) / 8, -((y - 1) * (z + 1)) / 8,
                      ((y + 1) * (z + 1)) / 8, -((y + 1) * (z + 1)) / 8],
                     [-((x - 1) * (z - 1)) / 8, ((x + 1) * (z - 1)) / 8, -((x + 1) * (z - 1)) / 8,
                      ((x - 1) * (z - 1)) / 8, ((x - 1) * (z + 1)) / 8, -((x + 1) * (z + 1)) / 8,
                      ((x + 1) * (z + 1)) / 8, -((x - 1) * (z + 1)) / 8],
                     [-((x - 1) * (y - 1)) / 8, ((x + 1) * (y - 1)) / 8, -((x + 1) * (y + 1)) / 8,
                      ((x - 1) * (y + 1)) / 8, ((x - 1) * (y - 1)) / 8, -((x + 1) * (y - 1)) / 8,
                      ((x + 1) * (y + 1)) / 8, -((x - 1) * (y + 1)) / 8]], dtype=C.dtype, device=device)

                J = qxqyqz @ dxdydz
                if torch.abs(J.det()) < 1e-12:
                     continue
                invJ = torch.inverse(J)
                qxyz = invJ @ qxqyqz
                B = torch.zeros(6, 24, dtype=C.dtype, device=device)

                for i_B in range(8):
                    B[:, i_B * 3:(i_B + 1) * 3] = torch.as_tensor(
                        [[qxyz[0, i_B], 0, 0],
                         [0, qxyz[1, i_B], 0],
                         [0, 0, qxyz[2, i_B]],
                         [qxyz[1, i_B], qxyz[0, i_B], 0],
                         [0, qxyz[2, i_B], qxyz[1, i_B]],
                         [qxyz[2, i_B], 0, qxyz[0, i_B]]], dtype=C.dtype, device=device)

                weight = J.det() * ww[i] * ww[j] * ww[k]
                Ke = Ke + weight * B.transpose(0, 1) @ C @ B
                Fe = Fe + weight * B.transpose(0, 1) @ C
    return Ke, Fe

def voxel_XKF(elastic_tensor, r):

    device = elastic_tensor.device
    idx = torch.ones(24, dtype=torch.bool, device=device)

    idx[[0, 1, 2, 4, 5, 11]] = False
    ke, fe = hexahedron(elastic_tensor, r)
    X0 = torch.zeros(24, 6, dtype=elastic_tensor.dtype, device=device)

    X0[idx, :] = torch.inverse(ke[idx, :][:, idx]) @ fe[idx, :]
    return X0, ke, fe


def get_trilinear_prolongation_matrix(device, dtype):
    

    fine_coords = torch.tensor([[x, y, z] for x in range(3) for y in range(3) for z in range(3)], 
                               dtype=dtype, device=device)
    rst = fine_coords - 1.0 

    node_signs = torch.tensor([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                               [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]], dtype=dtype, device=device)

    N_mat = torch.zeros(27, 8, dtype=dtype, device=device)
    for i in range(8):
        ri, si, ti = node_signs[i]
        N_mat[:, i] = 0.125 * (1 + rst[:,0]*ri) * (1 + rst[:,1]*si) * (1 + rst[:,2]*ti)
        

    P_loc = torch.zeros(81, 24, dtype=dtype, device=device)
    for n_fine in range(27):
        for n_coarse in range(8):
            val = N_mat[n_fine, n_coarse]
            P_loc[n_fine*3:(n_fine+1)*3, n_coarse*3:(n_coarse+1)*3] = torch.eye(3, dtype=dtype, device=device) * val
    return P_loc

def sparse_coarsen_ke(fine_ke, fine_elem_coords):

    device = fine_ke.device
    dtype = fine_ke.dtype

    if fine_ke.numel() == 0:

        out_coords_dim = fine_elem_coords.shape[1]
        return fine_ke.new_zeros((0, 24, 24)), fine_elem_coords.new_zeros((0, out_coords_dim))

    fine_elem_coords = fine_elem_coords.to(device=device, dtype=torch.long)

    # -----------------------------
    # -----------------------------
    if fine_elem_coords.shape[1] == 4:
        b = fine_elem_coords[:, 0:1]          # (N,1)
        xyz = fine_elem_coords[:, 1:4]        # (N,3)
        coarse_xyz = xyz // 2
        remainders = xyz % 2
        coarse_coords = torch.cat([b, coarse_xyz], dim=1)  # (N,4)
    else:
        xyz = fine_elem_coords                # (N,3)
        coarse_coords = xyz // 2
        remainders = xyz % 2


    local_indices = remainders[:, 0] * 4 + remainders[:, 1] * 2 + remainders[:, 2]


    unique_coarse, inverse_idx = torch.unique(coarse_coords, dim=0, return_inverse=True)
    M = unique_coarse.shape[0]

    # -----------------------------

    # -----------------------------
    ke_grouped = fine_ke.new_zeros((M, 8, 24, 24))
    for i in range(8):
        mask = (local_indices == i)
        if mask.any():
            coarse_ids = inverse_idx[mask]          # (n_i,)
            ke_grouped[:, i].index_add_(0, coarse_ids, fine_ke[mask])

    # -----------------------------
    # -----------------------------
    K_patch = fine_ke.new_zeros((M, 81, 81))


    child_offsets = torch.tensor(
        [[0, 0, 0],
         [0, 0, 1],
         [0, 1, 0],
         [0, 1, 1],
         [1, 0, 0],
         [1, 0, 1],
         [1, 1, 0],
         [1, 1, 1]],
        device=device, dtype=torch.long
    )


    node_offsets = torch.tensor(
        [[0, 0, 0],
         [1, 0, 0],
         [1, 1, 0],
         [0, 1, 0],
         [0, 0, 1],
         [1, 0, 1],
         [1, 1, 1],
         [0, 1, 1]],
        device=device, dtype=torch.long
    )


    xyz_child = child_offsets.unsqueeze(1) + node_offsets.unsqueeze(0)    # (8,8,3)
    patch_node_id = 9 * xyz_child[..., 0] + 3 * xyz_child[..., 1] + xyz_child[..., 2]  # (8,8)
    dof_base = patch_node_id.unsqueeze(-1) * 3                             # (8,8,1)
    dof_idx = (dof_base + torch.arange(3, device=device, dtype=torch.long).view(1, 1, 3)).reshape(8, 24)  # (8,24)


    for i in range(8):
        sub_ke = ke_grouped[:, i]  # (M,24,24)

        if torch.sum(torch.abs(sub_ke)) == 0:
            continue

        di = dof_idx[i]  # (24,)
        rows = di.view(-1, 1).repeat(1, 24)  # (24,24)
        cols = di.view(1, -1).repeat(24, 1)  # (24,24)
        K_patch[:, rows, cols] += sub_ke

    # -----------------------------

    # -----------------------------
    P = get_trilinear_prolongation_matrix(device, dtype)   # (81,24)
    coarse_ke = torch.matmul(P.t(), torch.matmul(K_patch, P))  # (M,24,24)

    return coarse_ke, unique_coarse



class EBEOperator(nn.Module):
    def __init__(self):
        super().__init__()


    def _compute_8_colors(self, coords):
        coords = coords.long()
        if coords.shape[1] == 4:
            coords = coords[:, 1:]
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    
        color_ids = (x % 2) * 4 + (y % 2) * 2 + (z % 2)
    
        color_groups = []
        for c in range(8):
            color_groups.append(torch.where(color_ids == c)[0])
        return color_groups
    def _compute_diag_inverse(self, Ke, num_nodes, dof_indices, fixed_mask=None):
        """Compute inverse of the assembled diagonal (per DOF).

        fixed_mask: (num_nodes,3) bool -> those DOFs get diag=1 (so inv=1) and are not updated.
        """
        device = Ke.device
        dtype = Ke.dtype

        elem_diag = torch.diagonal(Ke, dim1=-2, dim2=-1)  # (E,24)
        global_diag = torch.zeros((num_nodes * 3,), device=device, dtype=dtype)
        global_diag.index_add_(0, dof_indices.reshape(-1), elem_diag.reshape(-1))

        # Avoid zeros (unconnected dofs)
        global_diag[global_diag == 0] = 1.0

        if fixed_mask is not None:
            global_diag[fixed_mask.reshape(-1)] = 1.0

        return 1.0 / global_diag
    def matvec(self, Ke, u, dof_indices):
        """Matrix-free global matvec using element-by-element (EBE) assembly."""
        out_dtype = u.dtype
        if u.dtype != Ke.dtype:
            u_work = u.to(dtype=Ke.dtype)
        else:
            u_work = u

        u_flat = u_work.reshape(-1)                              # (nNodes*3,)
        u_elem = u_flat[dof_indices]                             # (E,24)
        f_elem = torch.bmm(Ke, u_elem.unsqueeze(-1)).squeeze(-1)  # (E,24)

        v_flat = torch.zeros_like(u_flat).to(f_elem.dtype)
        v_flat.index_add_(0, dof_indices.reshape(-1), f_elem.reshape(-1))
        v = v_flat.reshape(-1, 3)

        if v.dtype != out_dtype:
            v = v.to(dtype=out_dtype)
        return v
    def smooth_jacobi(self, Ke, u, f,node_coords, dof_indices, fixed_mask=None, omega=0.67, iterations=1, diag_inv=None):
        """Damped Jacobi smoother (optional)."""
        num_nodes = node_coords.shape[0]
        out_dtype = u.dtype
        work_dtype = Ke.dtype
        u_work = u.to(dtype=work_dtype)
        f_work = f.to(dtype=work_dtype)

        if diag_inv is None:
            diag_inv = self._compute_diag_inverse(Ke, num_nodes, dof_indices, fixed_mask=fixed_mask)
        diag_inv_reshaped = diag_inv.view(-1, 3)

        for _ in range(iterations):
            r = f_work - self.matvec(Ke, u_work, dof_indices)
            u_work = u_work + omega * diag_inv_reshaped * r

        return u_work.to(dtype=out_dtype)
    def smooth_gs_8color(self, Ke, u, f, node_coords, dof_indices, color_indices, diag_inv=None, fixed_mask=None, omega=1.0, iterations=1):
        """8-color Gauss-Seidel smoother on structured grid nodes."""
        out_dtype = u.dtype
        work_dtype = Ke.dtype
        u_work = u.to(dtype=work_dtype)
        f_work = f.to(dtype=work_dtype)

        num_nodes = node_coords.shape[0]
        if diag_inv is None:
            diag_inv = self._compute_diag_inverse(Ke, num_nodes, dof_indices, fixed_mask=fixed_mask)
        diag_inv_reshaped = diag_inv.view(-1, 3)

        for _ in range(iterations):
            for color_c in range(8):
                nodes_c = color_indices[color_c]
                if nodes_c.numel() == 0:
                    continue

                r = f_work - self.matvec(Ke, u_work, dof_indices)
                r_c = r[nodes_c]

                u_work[nodes_c] += omega * diag_inv_reshaped[nodes_c] * r_c

        return u_work.to(dtype=out_dtype)


# ==========================================

# ==========================================

class GMGSolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.P_op = ProlongStencil()
    def _generate_next_level_topology(self, elem_coords: torch.Tensor):
        device = elem_coords.device

        # 8 corner offsets for a voxel element (x,y,z)
        offsets = torch.tensor([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ], device=device, dtype=elem_coords.dtype)

        if elem_coords.numel() == 0:
            topo_indices = torch.empty((0, 8), device=device, dtype=torch.long)

            out_dim = elem_coords.shape[1]
            unique_nodes = torch.empty((0, out_dim), device=device, dtype=elem_coords.dtype)
            return topo_indices, unique_nodes

        if elem_coords.shape[1] == 4:
            # --- batched: elem_coords = [b,x,y,z] ---
            b = elem_coords[:, 0:1].to(device=device)          # (E,1)
            xyz = elem_coords[:, 1:4].to(device=device)        # (E,3)

            dims = xyz.max(dim=0).values + 1                   # (3,)
            dims = torch.clamp(dims, min=1)


            all_xyz = (xyz.unsqueeze(1) + offsets.unsqueeze(0)) % dims.view(1, 1, 3)  # (E,8,3)
            b_rep = b.unsqueeze(1).expand(-1, 8, 1)                                   # (E,8,1)
            all_node_coords = torch.cat([b_rep, all_xyz], dim=2)                      # (E,8,4)

            flat = all_node_coords.reshape(-1, 4)                                     # (E*8,4)
            unique_nodes, inverse = torch.unique(flat, dim=0, return_inverse=True)
            topo_indices = inverse.reshape(-1, 8).to(torch.long)
            return topo_indices, unique_nodes

        else:
            # --- unbatched: elem_coords = [x,y,z] ---
            xyz = elem_coords.to(device=device)
            dims = xyz.max(dim=0).values + 1
            dims = torch.clamp(dims, min=1)

            all_node_coords = (xyz.unsqueeze(1) + offsets.unsqueeze(0)) % dims.view(1, 1, 3)  # (E,8,3)
            flat = all_node_coords.reshape(-1, 3)

            unique_nodes, inverse = torch.unique(flat, dim=0, return_inverse=True)
            topo_indices = inverse.reshape(-1, 8).to(torch.long)
            return topo_indices, unique_nodes


    def _build_prolongation_operator_robust(self, fine_nodes, coarse_nodes):

        device = fine_nodes.device
        dtype = torch.float
    
        fine_nodes = fine_nodes.to(device=device, dtype=torch.long)
        coarse_nodes = coarse_nodes.to(device=device, dtype=torch.long)
    
        # --- batched / unbatched split ---
        if fine_nodes.shape[1] == 4 or coarse_nodes.shape[1] == 4:
            if fine_nodes.shape[1] == 3:
                fine_nodes = torch.cat([torch.zeros((fine_nodes.shape[0], 1), device=device, dtype=torch.long), fine_nodes], dim=1)
            if coarse_nodes.shape[1] == 3:
                coarse_nodes = torch.cat([torch.zeros((coarse_nodes.shape[0], 1), device=device, dtype=torch.long), coarse_nodes], dim=1)
    
            bF = fine_nodes[:, 0]
            xyzF = fine_nodes[:, 1:]
            bC = coarse_nodes[:, 0]
            xyzC = coarse_nodes[:, 1:]
    
            nF = xyzF.shape[0]
            nC = xyzC.shape[0]
    
            coarse_dims = (xyzC.max(dim=0).values + 1).clamp(min=1)  # (Dx,Dy,Dz)
            Dx, Dy, Dz = [int(x.item()) for x in coarse_dims]
            stride_yz = Dy * Dz
            stride_b = Dx * Dy * Dz
    
            # coarse linear keys
            linC = bC * stride_b + xyzC[:, 0] * stride_yz + xyzC[:, 1] * Dz + xyzC[:, 2]
            sort_idx = torch.argsort(linC)
            linC_sorted = linC[sort_idx]
    
            base = (xyzF // 2) % coarse_dims.view(1, 3)
            remainder = xyzF % 2  # (nF,3)
    
            offsets8 = torch.tensor(
                [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                 [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
                device=device, dtype=torch.long
            )
    
            c_idx = torch.full((nF, 8), -1, device=device, dtype=torch.long)
            w = torch.zeros((nF, 8), device=device, dtype=dtype)
    
            for k in range(8):
                off = offsets8[k].view(1, 3)
                target_xyz = (base + off) % coarse_dims.view(1, 3)
                linT = bF * stride_b + target_xyz[:, 0] * stride_yz + target_xyz[:, 1] * Dz + target_xyz[:, 2]
    
                pos = torch.searchsorted(linC_sorted, linT)
                valid = pos < linC_sorted.numel()
                pos_v = pos[valid]
                linT_v = linT[valid]
                ok = linC_sorted[pos_v] == linT_v
                if ok.any():
                    fine_rows = torch.nonzero(valid, as_tuple=False).squeeze(1)[ok]
                    coarse_rows = sort_idx[pos_v[ok]]
    
                    d_vec = torch.abs(0.5 * remainder[fine_rows].to(torch.float32) - off.to(torch.float32))
                    w_k = (1.0 - d_vec).prod(dim=1).to(dtype)
    
                    c_idx[fine_rows, k] = coarse_rows
                    w[fine_rows, k] = w_k
    
            return (c_idx, w)
    
        # --- original unbatched path (coords are (N,3)) ---
        nF = fine_nodes.shape[0]
    
        fine_dims = (fine_nodes.max(dim=0).values + 1).clamp(min=1)
        coarse_dims = (coarse_nodes.max(dim=0).values + 1).clamp(min=1)
    
        coarse_lookup = torch.full(
            (int(coarse_dims[0].item()), int(coarse_dims[1].item()), int(coarse_dims[2].item())),
            -1, device=device, dtype=torch.long
        )
        coarse_lookup[coarse_nodes[:, 0], coarse_nodes[:, 1], coarse_nodes[:, 2]] = torch.arange(coarse_nodes.shape[0], device=device, dtype=torch.long)
    
        base_coarse = fine_nodes // 2
        remainder = fine_nodes % 2  # 0/1
    
        offsets8 = torch.tensor(
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
             [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
            device=device, dtype=torch.long
        )
    
        c_idx = torch.full((nF, 8), -1, device=device, dtype=torch.long)
        w = torch.zeros((nF, 8), device=device, dtype=dtype)
    
        for k in range(8):
            off = offsets8[k].view(1, 3)
            target = (base_coarse + off) % coarse_dims.view(1, 3)  # (nF,3)
            c_k = coarse_lookup[target[:, 0], target[:, 1], target[:, 2]]  # (nF,)
    
            d_vec = torch.abs(0.5 * remainder.to(torch.float32) - off.to(torch.float32))  # (nF,3)
            w_k = (1.0 - d_vec).prod(dim=1).to(dtype)  # (nF,)
    
            ok = c_k >= 0
            if ok.any():
                c_idx[ok, k] = c_k[ok]
                w[ok, k] = w_k[ok]
    
        return (c_idx, w)
    def _spconv_to_dense_u(self, sp_tensor, op_levels,level: int, batch_idx = None, feat_slice=slice(0, 3)):

        device = sp_tensor.features.device
        dtype = sp_tensor.features.dtype
        node_coords = op_levels[level]['node_coords'].to(device=device, dtype=torch.long)
        n_nodes = node_coords.shape[0]
        u0 = torch.zeros((n_nodes, 3), device=device, dtype=dtype)
    
        if sp_tensor is None:
            return u0
    
        idx = getattr(sp_tensor, "indices", None)
        feat = getattr(sp_tensor, "features", None)
    
        if idx.numel() == 0:
            return u0
    
        idx = idx.to(device=device, dtype=torch.long)
        vals = feat.to(device=device)
    
        # --- parse node coords ---
        if node_coords.shape[1] == 4:
            b_nodes = node_coords[:, 0]
            xyz_nodes = node_coords[:, 1:]
        else:
            b_nodes = torch.zeros((n_nodes,), device=device, dtype=torch.long)
            xyz_nodes = node_coords
    
        # --- parse sparse coords ---
        if idx.shape[1] == 4:
            b_sp = idx[:, 0]
            xyz_sp = idx[:, 1:]
        else:
            b_sp = torch.zeros((idx.shape[0],), device=device, dtype=torch.long)
            xyz_sp = idx
    
        if batch_idx is not None:
            keep = (b_sp == int(batch_idx))
            b_sp = b_sp[keep]
            xyz_sp = xyz_sp[keep]
            vals = vals[keep]
    
        vals = vals[:, feat_slice]

        if vals.shape[1] > 3:
            vals = vals[:, :3]
    
        dims = (xyz_nodes.max(dim=0).values + 1).clamp(min=1)  # (Dx,Dy,Dz)
        xyz_sp = xyz_sp % dims.view(1, 3)
    
        Dx, Dy, Dz = [int(x.item()) for x in dims]
        stride_yz = Dy * Dz
        stride_b = Dx * Dy * Dz
    
        lin_nodes = b_nodes * stride_b + xyz_nodes[:, 0] * stride_yz + xyz_nodes[:, 1] * Dz + xyz_nodes[:, 2]
        lin_sp = b_sp * stride_b + xyz_sp[:, 0] * stride_yz + xyz_sp[:, 1] * Dz + xyz_sp[:, 2]
    
        sort_idx = torch.argsort(lin_nodes)
        lin_sorted = lin_nodes[sort_idx]
    
        pos = torch.searchsorted(lin_sorted, lin_sp)
        valid = (pos < lin_sorted.numel())
        pos = pos[valid]
        lin_sp_v = lin_sp[valid]
        valid2 = lin_sorted[pos] == lin_sp_v
        if not valid2.any():
            return u0
    
        rows = sort_idx[pos[valid2]]
        vals_v = vals[valid][valid2].to(device=device, dtype=dtype)
    
        u0.index_add_(0, rows, vals_v)
        return u0
    def _prepare_init_levels(self, init_list,op_levels, batch_idx = None, feat_slice=slice(0, 3)):

        if init_list is None:
            return None

    
        L = len(op_levels)
        init_levels = [None] * L
    
        for gmg_level in range(min(L, len(init_list))):
            sp_level = init_list[-1 - gmg_level]  # 
            init_levels[gmg_level] = self._spconv_to_dense_u(sp_level,op_levels, level=gmg_level, batch_idx=batch_idx, feat_slice=feat_slice)
    
        return init_levels
    def solve(
        self,
        fine_ke,
        global_f,
        node_coords_L0,
        fine_elem_coords,
        fine_topo_indices,
        max_iter: int = 50,
        smooth_iter = 1,
        init_list=None,
        init_batch=None,
        init_feat_slice=slice(0, 3),
        init_is_solution: bool = False,
        max_levels: int = 5,
        fixed_mask=None,
        dtype = torch.float
    ):

        device = fine_ke.device
        # dtype = fine_ke.dtype

        if global_f.dtype != dtype:
            global_f = global_f.to(dtype=dtype, device=device)

        op_levels = []

        # --- Level 0 (Finest) ---
        num_nodes_L0 = global_f.shape[0]
        op_L0 = EBEOperator()

        dof_indices_L0 = (fine_topo_indices.unsqueeze(-1) * 3 + torch.arange(3, device=device).reshape(1, 1, 3)).reshape(
            fine_topo_indices.shape[0], -1
        )

        op_levels.append(
            {
                "op": op_L0,
                "node_coords": node_coords_L0,
                "elem_coords": fine_elem_coords,
                "P": None,
                "Ke": fine_ke,
                "dof_indices": dof_indices_L0,
                "diag_inv": op_L0._compute_diag_inverse(fine_ke, num_nodes_L0, dof_indices_L0, fixed_mask=fixed_mask),
                "color_indices": op_L0._compute_8_colors(node_coords_L0),
            }
        )

        current_ke = fine_ke
        current_elem_coords = fine_elem_coords
        current_node_coords = node_coords_L0

        # --- Build Coarser Levels ---
        for l in range(1, max_levels):
            next_ke, next_elem_coords = sparse_coarsen_ke(current_ke, current_elem_coords)
            if next_ke.shape[0] == 0:
                break

            next_topo_indices, next_node_coords = self._generate_next_level_topology(next_elem_coords)
            num_nodes_next = next_node_coords.shape[0]

            op_next = EBEOperator()
            dof_indices_next = (next_topo_indices.unsqueeze(-1) * 3 + torch.arange(3, device=device).reshape(1, 1, 3)).reshape(
                next_topo_indices.shape[0], -1
            )

            op_levels.append(
                {
                    "op": op_next,
                    "node_coords": next_node_coords,
                    "elem_coords": next_elem_coords,
                    "P": self._build_prolongation_operator_robust(current_node_coords, next_node_coords),
                    "Ke": next_ke,
                    "dof_indices": dof_indices_next,
                    "diag_inv": op_next._compute_diag_inverse(next_ke, num_nodes_next, dof_indices_next, fixed_mask=None),
                    "color_indices": op_next._compute_8_colors(next_node_coords),
                }
            )

            # print(f"  Level {l}: {next_ke.shape[0]} Elements, {num_nodes_next} Nodes")

            current_ke = next_ke
            current_elem_coords = next_elem_coords
            current_node_coords = next_node_coords

        # --- Prepare init_levels (network predictions) ---
        u0_levels = self._prepare_init_levels(init_list, op_levels, batch_idx=init_batch, feat_slice=init_feat_slice)

        if u0_levels is not None and u0_levels[0] is not None:
            u = u0_levels[0]
            # .clone()
        else:
            u = torch.zeros_like(global_f)

        # enforce Dirichlet at level0
        if fixed_mask is not None:
            u[fixed_mask] = 0.0

        start_time = time.time()
        for it in range(max_iter):
            u = self._v_cycle(
                0,
                u,
                global_f,
                u0_levels=u0_levels,
                op_levels=op_levels,
                smooth_iter = smooth_iter,
                init_is_solution=init_is_solution,
                fixed_mask_L0=fixed_mask,
            )


            r = global_f - op_levels[0]["op"].matvec(op_levels[0]["Ke"], u, op_levels[0]["dof_indices"])

        return u, r
    def _v_cycle(self, level, u, f, u0_levels=None, op_levels=None, smooth_iter = 1, init_is_solution: bool = True, fixed_mask_L0=None):
        op = op_levels[level]["op"]
        lvl = op_levels[level]

        # only enforce Dirichlet on finest level
        fixed = fixed_mask_L0 if level == 0 else None

        # 1. Pre-Smoothing
        if isinstance(smooth_iter, (list, tuple, ListConfig)):
            if len(smooth_iter) == 0:
                iter_smooth = 1
            elif level < len(smooth_iter):
                iter_smooth = smooth_iter[level]
            else:
                iter_smooth = smooth_iter[-1]
        else:
            iter_smooth = smooth_iter
        # print (iter_smooth,isinstance(smooth_iter, (list, tuple)),smooth_iter)
        # exit()
        u = op.smooth_gs_8color(
            lvl["Ke"],
            u,
            f,
            lvl["node_coords"],
            lvl["dof_indices"],
            lvl["color_indices"],
            diag_inv=lvl["diag_inv"],
            fixed_mask=fixed,
            iterations=iter_smooth,
        )

        # Coarsest Level Solve (a few smoother steps)
        if level == len(op_levels) - 1:
            return op.smooth_gs_8color(
                lvl["Ke"],
                u,
                f,
                lvl["node_coords"],
                lvl["dof_indices"],
                lvl["color_indices"],
                diag_inv=lvl["diag_inv"],
                fixed_mask=fixed,
                iterations=1,
            )

        # 2. Residual
        r = f - op.matvec(lvl["Ke"], u, lvl["dof_indices"])
        if fixed is not None:
            r = r.clone()
            r[fixed] = 0.0

        # 3. Restriction (P^T)
        c_idx, w = op_levels[level + 1]["P"]
        nC = op_levels[level + 1]["node_coords"].shape[0]
        r_coarse = self.P_op.restrict_PT(r, nC=nC, c_idx=c_idx, w=w)

        # --- Inject coarse init from network (already dense-aligned) ---
        if u0_levels is not None and (level + 1) < len(u0_levels) and u0_levels[level + 1] is not None:
            u_pred_coarse = u0_levels[level + 1].to(device=r_coarse.device, dtype=r_coarse.dtype)
            if init_is_solution:
                # e0 = u_pred - R(u_current)
                u_restricted = self.P_op.restrict_PT(u, nC=nC, c_idx=c_idx, w=w).to(device=r_coarse.device, dtype=r_coarse.dtype)
                u_coarse = u_pred_coarse - u_restricted
            else:
                u_coarse = u_pred_coarse
        else:
            u_coarse = torch.zeros_like(r_coarse)

        # 4. Recursion
        u_coarse = self._v_cycle(
            level + 1,
            u_coarse,
            r_coarse,
            u0_levels=u0_levels,
            op_levels=op_levels,
            init_is_solution=init_is_solution,
            fixed_mask_L0=None,
        )

        # 5. Prolongation (P)
        correction = self.P_op.prolong(u_coarse, c_idx=c_idx, w=w)
        u_new = u + correction

        if fixed is not None:
            u_new[fixed] = 0.0

        # 6. Post-Smoothing
        u_new = op.smooth_gs_8color(
            lvl["Ke"],
            u_new,
            f,
            lvl["node_coords"],
            lvl["dof_indices"],
            lvl["color_indices"],
            diag_inv=lvl["diag_inv"],
            fixed_mask=fixed,
            iterations=iter_smooth,
        )
        return u_new


# ==========================================
# ==========================================

if __name__ == '__main__':
    
    
    # Just For Test

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")

    # -----------------------------
    # -----------------------------
    paths = [
        '00000.npz',
    ]

    coords_list = []
    topo_list = []
    voxel_list = []
    F_list = []  

    for p in paths:
        data = np.load(p, allow_pickle=True)
        node_coo  = torch.from_numpy(data['coords']).to(device)       # (nNodes,3)
        node_index = torch.from_numpy(data['node_index']).to(device)  # (nElem,8) 
        voxel     = torch.from_numpy(data['voxel']).to(device)        # (res,res,res)
        coords_list.append(node_coo)
        topo_list.append(node_index)
        voxel_list.append(voxel)


    batch_size = len(coords_list)
    res = int(voxel_list[0].shape[0])
    
    print(f"batch_size={batch_size}, res={res}")

    # -----------------------------

    # -----------------------------
    all_node_coords = []
    all_elem_coords = []
    all_topo = []


    node_offset = 0
    for b, (node_coo, topo, voxel) in enumerate(zip(coords_list, topo_list, voxel_list)):
        n_nodes = int(node_coo.shape[0])

        # node coords: (nNodes,4)
        bcol = torch.full((n_nodes, 1), b, dtype=torch.long, device=device)
        node_bxyz = torch.cat([bcol, node_coo.long()], dim=1)
        all_node_coords.append(node_bxyz)

        elem_xyz = torch.nonzero(voxel, as_tuple=False).long()  # (nElem,3)
        be = torch.full((elem_xyz.shape[0], 1), b, dtype=torch.long, device=device)
        elem_bxyz = torch.cat([be, elem_xyz], dim=1)
        all_elem_coords.append(elem_bxyz)

        topo = topo.long() + node_offset
        all_topo.append(topo)

        node_offset += n_nodes

    node_coords_batched = torch.cat(all_node_coords, dim=0)  # (Ntot,4)
    elem_coords_batched = torch.cat(all_elem_coords, dim=0)  # (Etot,4)
    topo_batched        = torch.cat(all_topo, dim=0)         # (Etot,8)

    print("node_coords_batched:", node_coords_batched.shape)
    print("elem_coords_batched:", elem_coords_batched.shape)
    print("topo_batched:", topo_batched.shape)

    # -----------------------------
    # -----------------------------
    C_H = isotropic_elastic_tensor(1.0, 0.3).to(device)
    _, Ke_elem, Fe = voxel_XKF(C_H, res)   # (24,24)
    E_tot = topo_batched.shape[0]
    Ke = Ke_elem.unsqueeze(0).repeat(E_tot, 1, 1).contiguous()

    f_list = [assembly_F(voxel, Fe) for voxel in voxel_list]
    F_dense_batched=torch.cat(f_list,dim=0)

    # -----------------------------
    # -----------------------------
    solver = GMGSolver()

    init_sp = spconv.SparseConvTensor(
        features=torch.zeros((node_coords_batched.shape[0], 3), device=device),
        indices=node_coords_batched.int(),
        spatial_shape=[res, res, res],
        batch_size=batch_size
    )


    # -----------------------------
    # -----------------------------
    u_sol = solver.solve(
        fine_ke = Ke,
        global_f=F_dense_batched,
        node_coords_L0 = node_coords_batched,
        fine_elem_coords = elem_coords_batched,
        fine_topo_indices = topo_batched,
        max_iter=110,
        max_levels=5,
        init_list=[init_sp],     
        init_batch=None,         
        init_feat_slice=slice(0, 3),
        init_is_solution=True
    )

    print(f"Max displacement: {torch.max(torch.abs(u_sol)):.6f}")
