import os
import torch
import random
import argparse
import numpy as np
from omegaconf import OmegaConf

from train.sp_lightning import *
from lightning.pytorch.callbacks import ModelCheckpoint

import lightning as pl

# torch.autograd.set_detect_anomaly(True)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')
torch.set_printoptions(
    precision=6,    
    threshold=200,
    edgeitems=100,
    linewidth=150,
    profile=None,
    sci_mode=True
)

def GMT_main(cfg):

    setup_seed(0)
    
    # Logger
    tb_logger = pl.pytorch.loggers.TensorBoardLogger(
                                                        save_dir=os.path.join(cfg.output_path,cfg.logger.path),
                                                        version=cfg.logger.version
                                                    )
    

    pl_module = LightningModule(cfg)  

    ckpt_callback= ModelCheckpoint(
        monitor="val_r", 
        dirpath=os.path.join(cfg.output_path,cfg.checkpoint.path),
        filename='epoch{epoch:02d}-residual{val_r:.6f}',
        save_last=True,
        save_top_k=1,
        auto_insert_metric_name=False,
        mode='min'
        )



    trainer = pl.Trainer(
                        max_epochs=cfg.max_epoch, 
                        check_val_every_n_epoch=1,
                        accelerator=cfg.device, 
                        strategy="ddp",
                        logger=tb_logger,
                        log_every_n_steps=50,
                        gradient_clip_val=1.0,
                        gradient_clip_algorithm="norm",
                        accumulate_grad_batches=4,
                        num_sanity_val_steps=1, 
                        callbacks=[ckpt_callback]
                        )
    
    dm = SparseDataModule(  train_data_path = cfg.train_data_path, 
                            val_data_path   = cfg.vail_data_path,
                            num_workers     = cfg.num_works,
                            batch_size      = cfg.batch_size)
    trainer.fit(pl_module,
                dm,
                ckpt_path=cfg.pre_train
            )

    



if __name__=='__main__':
    
    parser=argparse.ArgumentParser(description="GMT")

    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--world_size', type=int, default=4,help='Number of visible GPU')
    args=parser.parse_args()
    
    cfg=OmegaConf.load(args.config)
    GMT_main(cfg)