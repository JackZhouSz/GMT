
import torch
import torch.nn as nn
import numpy as np
import math
import spconv.pytorch as spconv
from train._utils import smooth_PCG_sp_assembly, EBE_PCG_chebyshev,EBE_PCG, PCG
from train.PTv3_3 import Point, PointTransformerV3
from train.EBE_GMG import GMGSolver
from torch_scatter import scatter_mean
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import time

class Encoder(nn.Module):
    def __init__(self, in_ch=16, out_ch=32, key='enc0'):
        super().__init__()
        self.encoder = spconv.SparseSequential(
            spconv.SubMConv3d(in_ch, out_ch, 3, padding=1, bias=False, indice_key=key),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        return self.encoder(x)



class Decoder(nn.Module):
    """
    Decoder block extracted from U-Net: three SubMConv3d layers + classifier
    """
    def __init__(self, in_ch, out_ch, key_prefix):
        super().__init__()
        self.dec = spconv.SparseSequential(
            spconv.SubMConv3d(in_ch, 32, 3, padding=1, bias=False, indice_key=key_prefix+'_1'),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(32, 16, 3, padding=1, bias=False, indice_key=key_prefix+'_2'),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(16, 8, 3, padding=1, bias=False, indice_key=key_prefix+'_3'),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(8, out_ch)
        self.tanh = nn.Tanh()

    def forward(self, x: spconv.SparseConvTensor):
        # print('decoder input',x.features.shape)
        out = self.dec(x)
        out = out.replace_feature(self.classifier(out.features))
        return out
        

    
class GMT(nn.Module):
    def __init__(self,cfg):
        super(GMT, self).__init__()
        self.cfg=cfg
        self.r=cfg.resolution
        self.sub_net=None
        self.encoder=Encoder(8,cfg.model.trans_channel[0])
        self.six = nn.ModuleList()
        for i in range(6):
            decoder = nn.ModuleList()
            for lvl, ch in enumerate(reversed(cfg.model.trans_channel)):
                decoder.append(Decoder(ch, 3, key_prefix=f"six{i}_lvl{lvl}"))  # Decoder  key_prefix
            self.six.append(decoder)
        self.gmg_solver=GMGSolver()
        self.ptv3 = PointTransformerV3(cfg)
    def sparse_to_point(self, x):
        return {
            'coord': x.indices[:,1:].float() ,
            'feat':  x.features,
            'batch': x.indices[:,0],
            'grid_size': 1,
            'sparse_shape' : [self.r,self.r,self.r]
        }

    def forward(self,batch,Ke,F,batch_size):
        param  = next(self.parameters())
        device = param.device
        dtype  = param.dtype
        F = F.reshape(-1,3,6)
        coo,feature=batch["coord"].int(),batch["feature"].to(dtype)
        # feature.requires_grad_(True)
        sparse_tensor = spconv.SparseConvTensor(
            features=feature,
            indices=coo,
            spatial_shape=[self.r,self.r,self.r],
            batch_size=batch_size
        )
        

        sparse_tensor = self.encoder(sparse_tensor)
        ptv3_input = self.sparse_to_point(sparse_tensor)
        spfea_list = self.ptv3(ptv3_input)
        

        E = batch['node_index'].shape[0]
        Ke = Ke.unsqueeze(0).expand(E, -1, -1)
        six_list_u ,six_list_r= [], []

        for i in range(6):
            decoder_list = self.six[i]
            init_list = [decoder(feat) for decoder, feat in zip(decoder_list, spfea_list)]
            u, r = self.gmg_solver.solve(
                                        fine_ke             = Ke,
                                        global_f            = F[:,:,i],
                                        node_coords_L0      = coo,
                                        fine_elem_coords    = batch['elem_coords'],
                                        fine_topo_indices   = batch['node_index'],
                                        max_iter            = self.cfg.GMG.max_cycle,
                                        smooth_iter         = self.cfg.GMG.smooth_iter,
                                        init_list           = init_list,
                                        max_levels          = len(init_list),
                                        dtype               = dtype
                                    )
            six_list_u.append(u)    
            six_list_r.append(r)

        out_u = torch.cat(six_list_u, dim=1)
        out_r = torch.cat(six_list_r, dim=1)

        return  out_u, out_r