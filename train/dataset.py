import torch
import os
import numpy as np
from tqdm import tqdm


class SparseDataset(torch.utils.data.Dataset):
    def __init__(self,base_path_base):
        self.data_name_set=[]
        for base_path in base_path_base:
            datanames = os.listdir(base_path)
            for dataname in tqdm(datanames, desc='Data Loading'):
                if dataname.lower().endswith('.npz'):
                    self.data_name_set.append(os.path.join(base_path,dataname))

    def __len__(self):
        return len(self.data_name_set)
    
    def __getitem__(self, index):
        data=np.load(self.data_name_set[index],allow_pickle=True)
        return torch.from_numpy(data['coords']), torch.from_numpy(data['node_type']), torch.from_numpy(data['node_index']), torch.from_numpy(data['voxel'])


def sparse_collate(batch):
    coords_list, feats_list, node_index_list, voxels = zip(*batch)
    batch_indices   = []
    all_node_index  = []
    all_coords      = []
    all_feats       = []
    voxel_batch_id  = []
    all_elem_coords = []
    node_base = 0
    for b_idx, (coords, feats,node_index,voxel) in enumerate(zip(coords_list, feats_list,node_index_list,voxels)):
        B = coords.shape[0]
        batch_idx = torch.full((B,1), b_idx, dtype=torch.long)
        voxel_batch_id.append(torch.full((node_index.size(0),1), b_idx, dtype=torch.long))
        idx_coords = torch.cat([batch_idx, coords.long()], dim=1)
        all_coords.append(idx_coords)
        all_feats.append(feats)
        all_node_index.append(node_index.long() + node_base)

        elem_xyz = torch.nonzero(voxel, as_tuple=False).long()  # (nElem,3)
        be = torch.full((elem_xyz.shape[0], 1), b_idx, dtype=torch.long)
        elem_bxyz = torch.cat([be, elem_xyz], dim=1)
        all_elem_coords.append(elem_bxyz)
        
        node_base += B

    all_indices     = torch.cat(all_coords, dim=0)      # (Ntot,4)
    elem_coords     = torch.cat(all_elem_coords, dim=0) # (Etot,4)
    all_node_index  = torch.cat(all_node_index, dim=0)  # (Etot,8)
    voxel_batch_id  = torch.cat(voxel_batch_id,dim=0)
    all_feats       = torch.cat(all_feats, dim=0)
    
    voxels = torch.stack(voxels, dim=0)
    return {
        "coord"         : all_indices,
        "feature"       : all_feats,
        "voxel"         : voxels,
        "node_index"    : all_node_index,
        "elem_coords"   : elem_coords, # (Etot,4)
        "voxel_batch_id": voxel_batch_id.squeeze()
    }