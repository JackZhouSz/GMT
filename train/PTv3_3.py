"""
Point Transformer - V3 Mode1
"""
import copy
import sys
from functools import partial
from addict import Dict
import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.models.layers import DropPath
from collections import OrderedDict

try:
    import flash_attn
except ImportError:
    flash_attn = None

from .serialization import encode


class RoPE(nn.Module):
    def __init__(self, feature_dim=8, head_num=4, base_res=64, D=3):
        super().__init__()
        assert feature_dim % (head_num * D) == 0
        self.feature_dim, self.head_num, self.base_res, self.D = feature_dim, head_num, base_res, D
        self.k_max = feature_dim // (2 * head_num * D)

        self.register_buffer(
            "freqs_base",
            (torch.arange(self.k_max, dtype=torch.float32) + 1.0) / float(self.base_res),
            persistent=False
        )

    def precompute_freqs(self, coo_1d: torch.Tensor, out_dtype: torch.dtype):
        coo_f = coo_1d.to(dtype=torch.float32)
        freqs = self.freqs_base.to(device=coo_1d.device)  # float32
        theta = torch.outer(coo_f, freqs)  # float32
        cos = torch.cos(theta).to(dtype=out_dtype)
        sin = torch.sin(theta).to(dtype=out_dtype)
        return cos, sin

    @staticmethod
    def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        xr = x[..., 0]
        xi = x[..., 1]
        out_r = xr * cos - xi * sin
        out_i = xr * sin + xi * cos
        return torch.stack((out_r, out_i), dim=-1)

    def forward(self, qkv: torch.Tensor, coo: torch.Tensor) -> torch.Tensor:
        node_num = qkv.shape[0]
        K = self.feature_dim // (self.head_num * self.D * 2)

        xq = qkv[:, 0:self.feature_dim].reshape(node_num, self.D, self.head_num, K, 2)
        xk = qkv[:, self.feature_dim:self.feature_dim*2].reshape(node_num, self.D, self.head_num, K, 2)
        xv = qkv[:, self.feature_dim*2:self.feature_dim*3]

        with torch.cuda.amp.autocast(enabled=False):
            out_dtype = xq.dtype
            cos_list, sin_list = [], []
            for i in range(self.D):
                cos_i, sin_i = self.precompute_freqs(coo[:, i], out_dtype=out_dtype)
                cos_list.append(cos_i)
                sin_list.append(sin_i)
            cos = torch.stack(cos_list, dim=1).unsqueeze(2)  # [N, D, 1, K]
            sin = torch.stack(sin_list, dim=1).unsqueeze(2)  # [N, D, 1, K]

        xq = self.apply_rope(xq, cos, sin).reshape(node_num, 1, -1)
        xk = self.apply_rope(xk, cos, sin).reshape(node_num, 1, -1)
        return torch.cat([xq, xk, xv.reshape(node_num, 1, -1)], dim=1).reshape(node_num, -1)


@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )

@torch.inference_mode()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)

@torch.inference_mode()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


# (kept for compatibility; no longer used in forward after modification)
def Pointneworder(point, order="x"):
    newpoint = Point({
        'coord':        point.coord.clone(),
        'feat':         point.feat.clone(),
        'batch':        point.batch.clone(),
        'grid_size':    point.grid_size,
        'sparse_shape': point.sparse_shape,
    })
    newpoint.serialization(order=order)
    newpoint.sparsify()
    return newpoint

def Point_dict_clone(point):
    newpoint = Point({
        'coord':        point.coord.clone(),
        'feat':         point.feat.clone(),
        'batch':        point.batch.clone(),
        'grid_size':    point.grid_size,
        'sparse_shape': point.sparse_shape,
    })
    newpoint.sparsify()
    return newpoint


class Point(Dict):
    """
    Point Structure of Pointcept
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            self["grid_coord"] = self.coord

        if depth is None:
            depth = int(self.grid_coord.max()).bit_length()
        self["serialized_depth"] = depth
        assert depth * 3 + len(self.offset).bit_length() <= 63
        assert depth <= 16

        # allow order be str or tuple/list of str
        if isinstance(order, str):
            order = (order,)

        code = [encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order]
        code = torch.stack(code)  # (k, n)
        order_idx = torch.argsort(code)  # (k, n)
        inverse = torch.zeros_like(order_idx).scatter_(
            dim=1,
            index=order_idx,
            src=torch.arange(0, code.shape[1], device=order_idx.device).repeat(code.shape[0], 1),
        )

        if shuffle_orders:
            perm = torch.randperm(code.shape[0], device=code.device)
            code = code[perm]
            order_idx = order_idx[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order_idx
        self["serialized_inverse"] = inverse

    def sparsify(self, pad=96):
        assert {"feat", "batch"}.issubset(self.keys())
        if "grid_coord" not in self.keys():
            self["grid_coord"] = self.coord

        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(torch.max(self.grid_coord, dim=0).values, pad).tolist()

        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat([self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=int(self.batch.max().item()) + 1 if self.batch.numel() > 0 else 1
        )
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat

    def synchronize(self):
        self.sparse_conv_feat = self.sparse_conv_feat.replace_feature(self.feat)


class PointModule(nn.Module):
    """placeholder, all module subclass from this will take Point in PointSequential."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointSequential(PointModule):
    """A sequential container that supports Point and SpConvTensor."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for _ in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for _, module in self._modules.items():
            if isinstance(module, PointModule):
                input = module(input)
            elif spconv.modules.is_spconv_module(module):
                if isinstance(input, Point):
                    input.sparse_conv_feat = module(input.sparse_conv_feat)
                    input.feat = input.sparse_conv_feat.features
                else:
                    input = module(input)
            else:
                if isinstance(input, Point):
                    input.feat = module(input.feat)
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(input.feat)
                elif isinstance(input, spconv.SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        input = input.replace_feature(module(input.features))
                else:
                    input = module(input)
        return input


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_flash = enable_flash

        if enable_flash:
            assert upcast_attention is False, "Set upcast_attention to False when enable Flash Attention"
            assert upcast_softmax is False, "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.rope = RoPE(feature_dim=self.channels, head_num=self.num_heads)
        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (pad_key not in point.keys() or unpad_key not in point.keys() or cu_seqlens_key not in point.keys()):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(bincount + self.patch_size - 1, self.patch_size, rounding_mode="trunc")
                * self.patch_size
            )
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1] - self.patch_size + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1] - 2 * self.patch_size + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        qkv = self.qkv(point.feat)
        qkv = self.rope(qkv, point.coord)
        qkv = qkv[order]

        feat = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv.half().reshape(-1, 3, H, C // H),
            cu_seqlens,
            max_seqlen=self.patch_size,
            dropout_p=self.attn_drop if self.training else 0,
            softmax_scale=self.scale,
        ).reshape(-1, C)

        feat = feat.to(qkv.dtype)
        feat = feat[inverse]
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)

        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class GMGPoolingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph_data = None

    def build_connectivity(self, point):
        x_sparse = point.sparse_conv_feat
        fine_indices = point.sparse_conv_feat.indices
        device = fine_indices.device
        N = fine_indices.shape[0]
        current_res = point.sparse_shape[0]
        coarse_res = current_res // 2

        batch_idx = fine_indices[:, 0:1]
        coords = fine_indices[:, 1:]

        base_coarse = coords // 2
        is_odd = (coords % 2)

        offsets = torch.tensor([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
        ], device=device)

        valid_mask = torch.all(is_odd.unsqueeze(1) >= offsets.unsqueeze(0), dim=2)

        fine_node_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, 8)
        valid_fine_node_idx = fine_node_idx[valid_mask]

        candidates_coarse_coords = base_coarse.unsqueeze(1) + offsets.unsqueeze(0)
        candidates_coarse_coords = candidates_coarse_coords % coarse_res

        batch_idx_expanded = batch_idx.unsqueeze(1).expand(N, 8, 1)
        full_coarse_candidates = torch.cat([batch_idx_expanded, candidates_coarse_coords], dim=2)

        valid_coarse_indices_raw = full_coarse_candidates[valid_mask]

        num_odd_dims = torch.sum(is_odd, dim=1).float()
        fine_degree = (2.0 ** num_odd_dims).unsqueeze(1)

        unique_coarse_indices, inverse_indices = torch.unique(
            valid_coarse_indices_raw, dim=0, return_inverse=True
        )

        num_coarse = unique_coarse_indices.shape[0]
        coarse_degree = torch.zeros(num_coarse, 1, device=device)
        coarse_degree.index_add_(0, inverse_indices, torch.ones(valid_fine_node_idx.shape[0], 1, device=device))

        return {
            'edge_index': torch.stack([valid_fine_node_idx, inverse_indices]),
            'fine_degree': fine_degree,
            'coarse_degree': coarse_degree,
            'unique_coarse_indices': unique_coarse_indices,
            'num_fine': N,
            'num_coarse': num_coarse,
            'fine_indices': fine_indices,
            'fine_spatial_shape': x_sparse.spatial_shape,
            'batch_size': x_sparse.batch_size
        }

    def pooling(self, point):
        point.synchronize()
        x_sparse = point.sparse_conv_feat
        fine_feats = x_sparse.features
        self.graph_data = self.build_connectivity(point)
        gd = self.graph_data
        fine_idx, coarse_idx = gd['edge_index']

        out_feats = torch.zeros(gd['num_coarse'], fine_feats.shape[1], device=fine_feats.device)
        out_feats.index_add_(0, coarse_idx, fine_feats[fine_idx])
        out_feats = out_feats / (gd['coarse_degree'] + 1e-12)

        coarse_res = [r // 2 for r in point.sparse_shape]

        point = Point({
            'coord': gd['unique_coarse_indices'][:, 1:].float(),
            'feat':  out_feats,
            'batch': gd['unique_coarse_indices'][:, 0].int(),
            'grid_size': 1,
            'sparse_shape': coarse_res
        })
        # NOTE: serialization will be called by backbone after pooling (multi-order)
        point.sparsify()
        return point

    def unpooling(self, point):
        point.synchronize()
        x_coarse_sparse = point.sparse_conv_feat
        gd = self.graph_data
        coarse_feats = x_coarse_sparse.features
        fine_idx, coarse_idx = gd['edge_index']

        source_feats = coarse_feats[coarse_idx]
        out_feats = torch.zeros(
            gd['num_fine'], coarse_feats.shape[1],
            dtype=source_feats.dtype, device=coarse_feats.device
        )
        out_feats.index_add_(0, fine_idx, source_feats)
        out_feats = out_feats / gd['fine_degree']

        point = Point({
            'coord': gd['fine_indices'][:, 1:].float(),
            'feat':  out_feats,
            'batch': gd['fine_indices'][:, 0].int(),
            'grid_size': 1,
            'sparse_shape': gd['fine_spatial_shape']
        })
        # NOTE: serialization will be called by backbone after unpooling (multi-order)
        point.sparsify()
        return point


class PointTransformerV3(PointModule):
    def __init__(
        self,
        cfg,
        order=("x", "y", "z"),
        enc_depths=(4, 4, 4, 4),
        enc_channels=(24, 48, 96, 192),
        enc_num_head=(4, 4, 4, 4),
        enc_patch_size=(256, 512, 1024, 1024),
        dec_depths=(4, 4, 4),
        dec_channels=(24, 48, 96),
        dec_num_head=(4, 4, 4),
        dec_patch_size=(1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
    ):
        super().__init__()
        enc_depths = cfg.model.trans_block
        enc_channels = cfg.model.trans_channel
        enc_num_head = cfg.model.trans_head
        enc_patch_size = cfg.model.trans_window_size
        dec_depths = cfg.model.trans_block[:-1]
        dec_channels = cfg.model.trans_channel[:-1]
        dec_num_head = cfg.model.trans_head[:-1]
        dec_patch_size = cfg.model.trans_window_size[:-1]

        self.num_stages = len(enc_depths)

        # CHANGED: ensure tuple orders
        self.order = (order,) if isinstance(order, str) else tuple(order)
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(enc_depths) 
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        ln_layer = nn.LayerNorm
        act_layer = nn.GELU

        enc_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]

        self.pool_module = nn.ModuleList([GMGPoolingLayer() for _ in range(self.num_stages - 1)])

        # encoder
        self.enc = []
        self.enc_mlp = []
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[sum(enc_depths[:s]) : sum(enc_depths[: s + 1])]
            enc = PointSequential()
            for i in range(enc_depths[s]):
                # CHANGED: cycle order_index by block id
                order_index = i % len(self.order)
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=order_index,
                        cpe_indice_key=f"stage{s}",
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            self.enc.append(enc)

            if s != self.num_stages - 1:
                self.enc_mlp.append(
                    PointSequential(
                        MLP(
                            in_channels=enc_channels[s],
                            hidden_channels=int(enc_channels[s] * mlp_ratio),
                            out_channels=enc_channels[s + 1],
                            act_layer=act_layer,
                            drop=proj_drop,
                        )
                    )
                )

        self.enc = nn.ModuleList(self.enc)
        self.enc_mlp = nn.ModuleList(self.enc_mlp)

        # decoder
        self.dec = []
        self.dec_mlp = []
        dec_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))]
        dec_channels = list(dec_channels) + [enc_channels[-1]]

        for s in reversed(range(self.num_stages - 1)):
            dec_drop_path_ = dec_drop_path[sum(dec_depths[:s]) : sum(dec_depths[: s + 1])]
            dec_drop_path_.reverse()

            dec = PointSequential()
            dec.add(
                MLP(
                    in_channels=dec_channels[s] * 2,
                    hidden_channels=int(dec_channels[s] * mlp_ratio),
                    out_channels=dec_channels[s],
                    act_layer=act_layer,
                    drop=proj_drop,
                )
            )

            for i in range(dec_depths[s]):
                # CHANGED: cycle order_index by block id
                order_index = i % len(self.order)
                dec.add(
                    Block(
                        channels=dec_channels[s],
                        num_heads=dec_num_head[s],
                        patch_size=dec_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=dec_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=order_index,
                        cpe_indice_key=f"stage{s}",
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )

            self.dec_mlp.append(
                PointSequential(
                    MLP(
                        in_channels=dec_channels[s + 1],
                        hidden_channels=int(dec_channels[s] * mlp_ratio),
                        out_channels=dec_channels[s],
                        act_layer=act_layer,
                        drop=proj_drop,
                    )
                )
            )
            self.dec.append(dec)

        self.dec_mlp = nn.ModuleList(self.dec_mlp)
        self.dec = nn.ModuleList(self.dec)

    def forward(self, data_dict):
        """
        data_dict should contain:
        - "feat"
        - "grid_coord" or ("coord" + "grid_size")
        - "offset" or "batch"
        """
        point = Point(data_dict)

        # CHANGED: one-shot multi-order serialization
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        down_sample_feat = []

        # encoder
        for s in range(self.num_stages):
            if s > 0:
                point = self.pool_module[s - 1].pooling(point)
                # CHANGED: re-serialize after pooling (point set changed)
                point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
                point = self.enc_mlp[s - 1](point)

            # CHANGED: run once (no x/y/z cloned branches)
            point = self.enc[s](point)

            if s < self.num_stages - 1:
                down_sample_feat.append(point.feat.clone())

        down_sample_feat = down_sample_feat[::-1]

        # keep original behavior (cloning sp_level); you can remove clones later if desired
        point.sparsify()
        sp_level = []
        sp_level.append(spconv.SparseConvTensor(
            features=point.sparse_conv_feat.features.clone(),
            indices=point.sparse_conv_feat.indices.clone(),
            spatial_shape=point.sparse_shape,
            batch_size=int(point.batch.max().item()) + 1 if point.batch.numel() > 0 else 1
        ))

        # decoder
        for s in range(self.num_stages - 1):
            point = self.dec_mlp[s](point)
            point = self.pool_module[self.num_stages - 2 - s].unpooling(point)

            # CHANGED: re-serialize after unpooling
            point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)

            # concat skip (original behavior)
            point.feat = torch.cat([point.feat, down_sample_feat[s]], dim=1)

            # CHANGED: run once (no x/y/z cloned branches)
            point = self.dec[s](point)

            point.sparsify()
            sp_level.append(spconv.SparseConvTensor(
                features=point.sparse_conv_feat.features.clone(),
                indices=point.sparse_conv_feat.indices.clone(),
                spatial_shape=point.sparse_shape,
                batch_size=int(point.batch.max().item()) + 1 if point.batch.numel() > 0 else 1
            ))

        return sp_level
