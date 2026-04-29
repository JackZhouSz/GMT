import os
import math
import torch
import lightning as pl

from train._utils import (
    hexahedron,
    assembly,
    assemble_F_fast,
    isotropic_elastic_tensor,
    PCG,
)
from train.dataset import sparse_collate, SparseDataset
from train.minimal_potential_energy_loss import MPE_loss, MPE_loss_func
from train.model import GMT
from train.eval import *
import torch.optim as optim
from torch.utils.data import DataLoader



class CosineAnnealingWarmupRestarts(optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        assert warmup_steps < first_cycle_steps
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        super().__init__(optimizer, last_epoch)
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.max_lr - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.step_in_cycle - self.warmup_steps)
                        / (self.cur_cycle_steps - self.warmup_steps)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle += 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle -= self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult**n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * (self.cycle_mult**n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for pg, lr in zip(self.optimizer.param_groups, self.get_lr()):
            pg["lr"] = lr


class ExponentialMovingAverage:
    def __init__(self, parameters, decay, use_num_updates=True):
        if not (0.0 <= decay <= 1.0):
            raise ValueError("Decay must be between 0 and 1")
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.detach().clone() for p in parameters if p.requires_grad]
        self.collected_params = None

    def to(self, device=None, dtype=None):
        for i in range(len(self.shadow_params)):
            self.shadow_params[i] = self.shadow_params[i].to(device=device, dtype=(dtype or self.shadow_params[i].dtype))
        return self

    def update(self, parameters):
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            params = [p for p in parameters if p.requires_grad]
            for i, (s, p) in enumerate(zip(self.shadow_params, params)):
                if s.device != p.device or s.dtype != p.dtype:
                    self.shadow_params[i] = s.to(device=p.device, dtype=p.dtype)
                    s = self.shadow_params[i]
                s.sub_(one_minus_decay * (s - p))

    def store(self, parameters):
        params = [p for p in parameters if p.requires_grad]
        with torch.no_grad():
            self.collected_params = [p.detach().clone() for p in params]

    def copy_to(self, parameters):
        params = [p for p in parameters if p.requires_grad]
        with torch.no_grad():
            for s, p in zip(self.shadow_params, params):
                p.data.copy_(s.data)

    def restore(self, parameters):
        if self.collected_params is None:
            return
        params = [p for p in parameters if p.requires_grad]
        with torch.no_grad():
            for c, p in zip(self.collected_params, params):
                p.data.copy_(c.data)


class SparseDataModule(pl.LightningDataModule):
    def __init__(self, train_data_path, val_data_path, batch_size=8, num_workers=1):
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_ds = SparseDataset(self.train_data_path)
        self.val_ds = SparseDataset(self.val_data_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=sparse_collate,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=sparse_collate,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=4 if self.num_workers > 0 else None,
        )


class LightningModule(pl.LightningModule):


    def __init__(self, cfg):
        super().__init__()
        self.model = GMT(cfg)

        self.lr = cfg.learning_rate
        self.batch_size = cfg.batch_size
        self.resolution = cfg.resolution

        self.C = isotropic_elastic_tensor(1.0, 0.3)
        Ke, Fe, X0 = hexahedron(r=self.resolution, C=self.C)
        self.register_buffer("Ke", Ke)
        self.register_buffer("Fe", Fe)
        self.register_buffer("X0", X0)

        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.999)
        self._last_lr = None

    def on_fit_start(self):
        p = next(self.model.parameters())
        self.ema.to(device=p.device, dtype=p.dtype)

    # ---- EMA only for validation ----
    def on_validation_epoch_start(self):
        self.ema.store(self.model.parameters())
        self.ema.copy_to(self.model.parameters())

    def on_validation_epoch_end(self):
        self.ema.restore(self.model.parameters())

    def training_step(self, batch, batch_idx):
        n_nodes = batch["feature"].shape[0]
        F = assemble_F_fast(batch["node_index"], self.Fe, n_nodes)

        u, r = self.model(batch, self.Ke, F, self.batch_size)
        r64 = r.double()
        r64 = r64.view(-1, 3, 6)

        
        loss = torch.log10(torch.linalg.norm(r64))

        loss = 0.0
        eq_log_sum = r64.new_zeros((6,))  

        for i in range(self.batch_size):
            mask = (batch["coord"][:, 0] == i)

            eq_norm = torch.linalg.norm(r64[mask], dim=(0, 1))  # (6,)
            eq_log = eq_norm
            eq_log = torch.log10(eq_norm + 1e-30)                      # (6,)

            loss = loss + eq_log.mean()
            eq_log_sum = eq_log_sum + eq_log

        loss = loss / self.batch_size

        eq_log_mean = eq_log_sum / self.batch_size

        self.log("train_r", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=False, batch_size=self.batch_size)

        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True, on_step=True, on_epoch=False, sync_dist=False , batch_size=self.batch_size)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema.update(self.model.parameters())

    def validation_step(self, batch, batch_idx):
        n_nodes = batch["feature"].shape[0]
        F = assemble_F_fast(batch["node_index"], self.Fe, n_nodes)
        u, r = self.model(batch, self.Ke, F, self.batch_size)

        r64 = r.double()
        r64 = r64.view(-1, 3, 6)

    
        loss = 0.0

        for i in range(self.batch_size):
            mask = (batch["coord"][:, 0] == i)

            eq_norm = torch.linalg.norm(r64[mask], dim=(0, 1))  # (6,)
            # eq_log = eq_norm

            loss = loss + eq_norm.mean()




        self.log("val_r", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=False, batch_size=self.batch_size)


        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True, on_step=True, on_epoch=False, sync_dist=False , batch_size=self.batch_size)

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler
            }
        