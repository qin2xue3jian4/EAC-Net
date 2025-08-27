import torch
import numpy as np
from typing import List, Union

from .load import LoaderWrapper
from .dataset import MixDataset

def get_loader(
    paths: Union[str, List[str]],
    mode: str = 'train',
    out_type: str = 'mixed',
    root_dir: str = None,
    frame_size: int = 3,
    atom_rcut: float = 5.0,
    atom_sel: int = 40,
    probe_size: int = 50,
    probe_rcut: float = 6.0,
    probe_sel: int = 100,
    num_workers: int = 0,
    epoch_size: int = 100,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device('cpu'),
    ngfs_str: str = None,
    exclude_keys: List[str] = None,
    local_rank: int = 0,
    world_size: int = 1,
    search_depth: int = 6,
    lazy_load: bool = False,
    base_seed: int = 0,
):
    out_probe = out_type != 'potential'
    dataset = MixDataset(
        paths,
        mode=mode,
        out_probe=out_probe,
        root_dir=root_dir,
        atom_cutoff=atom_rcut,
        atom_sel=atom_sel,
        probe_cutoff=probe_rcut,
        probe_sel=probe_sel,
        dtype=dtype,
        ngfs_str=ngfs_str,
        search_depth=search_depth,
        lazy_load=lazy_load,
        exclude_keys=exclude_keys,
    )
    
    frame_shuffle = mode == 'train'
    probe_shuffle = (mode == 'train' or (mode == 'test' and epoch_size != -1))

    loader = LoaderWrapper(
        dataset,
        frame_size=frame_size,
        num_workers=num_workers,
        epoch_size=epoch_size,
        probe_size=probe_size,
        frame_shuffle=frame_shuffle,
        probe_shuffle=probe_shuffle,
        base_seed=base_seed,
        device=device,
        local_rank=local_rank,
        world_size=world_size,
    )
    
    return loader