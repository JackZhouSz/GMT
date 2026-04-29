import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def get_node_type(voxel: torch.Tensor) -> torch.Tensor:
    r = voxel.shape[0]
    node_type = torch.zeros((r, r, r, 8), dtype=torch.uint8, device=voxel.device)
    hex8 = torch.tensor(
        [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],
        dtype=torch.int64,
        device=voxel.device
    )
    solid_voxel_coo = torch.nonzero(voxel, as_tuple=False)
    for i in range(8):
        coo = (solid_voxel_coo + hex8[i] + r) % r
        node_type[coo[:, 0], coo[:, 1], coo[:, 2], i] = 1
    return node_type


def load_voxel_from_file(path: Path, res: int) -> np.ndarray:
    suf = path.suffix.lower()
    if suf == ".npy":
        arr = np.load(path)
    elif suf == ".csv":
        arr = np.loadtxt(path, delimiter=",")
    elif suf == ".npz":
        arr = np.load(path)["voxel"]
    else:
        raise ValueError(f"Unsupported file type: {path}")

    arr = np.asarray(arr)
    if arr.shape != (res, res, res):
        arr = arr.reshape(res, res, res)

    return (arr > 0)  # bool


@torch.no_grad()
def process_one_voxel(voxel_np: np.ndarray, res: int, device: torch.device):
    voxel = torch.from_numpy(voxel_np.astype(np.uint8)).to(device).reshape(res, res, res)
    voxel_coo = torch.nonzero(voxel, as_tuple=False)  # (Nv,3)

    node_full_grid = voxel.clone()
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                coo = (voxel_coo + torch.tensor([i, j, k], device=device) + res) % res
                node_full_grid[coo[:, 0], coo[:, 1], coo[:, 2]] = 1
    all_node_coo = torch.nonzero(node_full_grid, as_tuple=False)  # (Nn,3)

    node_type_grid = get_node_type(voxel)
    node_type = node_type_grid[all_node_coo[:, 0], all_node_coo[:, 1], all_node_coo[:, 2]]  # (Nn,8)

    node_dof_full_grid = torch.zeros_like(voxel, dtype=torch.int32)
    dof_index = torch.arange(all_node_coo.shape[0], dtype=torch.int32, device=device)
    node_dof_full_grid[all_node_coo[:, 0], all_node_coo[:, 1], all_node_coo[:, 2]] = dof_index

    node_index = torch.zeros((voxel_coo.shape[0], 8), dtype=torch.int32, device=device)
    hex8 = torch.tensor(
        [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],
        dtype=torch.int64,
        device=device
    )
    for i in range(8):
        coo = (hex8[i] + voxel_coo + res) % res
        node_index[:, i] = node_dof_full_grid[coo[:, 0], coo[:, 1], coo[:, 2]]

    return (
        all_node_coo.cpu().numpy().astype(np.int32),
        node_type.cpu().numpy().astype(np.uint8),
        voxel.cpu().numpy().astype(bool),
        node_index.cpu().numpy().astype(np.int32),
    )


def unique_path_if_exists(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suf = path.stem, path.suffix
    i = 1
    while True:
        cand = path.with_name(f"{stem}_{i}{suf}")
        if not cand.exists():
            return cand
        i += 1


def folder_id(idx: int) -> str:
    """
    0->A, 1->B, ..., 25->Z, 26->AA, 27->AB ...
    """
    s = ""
    idx += 1
    while idx > 0:
        idx, r = divmod(idx - 1, 26)
        s = chr(ord("A") + r) + s
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--res", type=int, default=64)
    parser.add_argument("--type", type=str, default="Truss")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--avoid_collision", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed, skipped = 0, 0

    for dir_idx, in_dir in enumerate(args.input_dirs):
        in_dir = Path(in_dir)
        if not in_dir.exists():
            print(f"[WARN] input dir not found: {in_dir}")
            continue

        dir_tag = folder_id(dir_idx)  # A/B/C...

        files = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in (".npy", ".csv", ".npz")]
        for path in tqdm(files, desc=f"Processing {in_dir.name} ({dir_tag})", leave=False):
 
            out_path = out_dir / f"{args.type}_{dir_tag}_{path.stem}.npz"

            if out_path.exists() and (not args.overwrite) and (not args.avoid_collision):
                skipped += 1
                continue
            if args.avoid_collision and (not args.overwrite):
                out_path = unique_path_if_exists(out_path)

            try:
                voxel_np = load_voxel_from_file(path, args.res)
                coords, node_type, voxel, node_index = process_one_voxel(voxel_np, args.res, device)

                np.savez(
                    out_path,
                    coords=coords,
                    node_type=node_type,
                    voxel=voxel,
                    node_index=node_index,
                )
                processed += 1
            except Exception as e:
                print(f"[ERROR] failed on {path}: {e}")

    print(f"Done. Processed={processed}, Skipped={skipped}, Out={out_dir}")


if __name__ == "__main__":
    main()
