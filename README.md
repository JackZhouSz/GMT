# GMT (SIGGRAPH 2026 Submission) — Code Release

This repository contains the official implementation accompanying the SIGGRAPH submission, for predicting the effective elasticity tensor \(C^H\). It includes training and evaluation code, with the default experiment configuration provided in `configs/train.yaml`.

## 1. Requirements

- Linux recommended
- NVIDIA GPU with CUDA support (NVIDIA RTX 5090 GPU (32 GB) in our experiments)
- Conda (Miniconda/Anaconda)

> Notes  
> - The full software environment is captured in `environment.yml`.  
> - Training runtime and memory usage depend on the voxel resolution, batch size, and model settings in `configs/train.yaml`.

## 2. Installation (Conda)

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate GMT
```

## 3. Configuration

 Configuration file: `configs/train.yaml`

## 4. Running
Run training with GPU 0:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py configs/train.yaml
```
Run training with multi-GPUs:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py configs/train.yaml
```

## 5. Outputs
Outputs are controlled by configs/train.yaml. Typical artifacts include:

- Model checkpoints (saved weights)
- Training logs (TensorBoard event files)