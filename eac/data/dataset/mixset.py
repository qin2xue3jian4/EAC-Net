import math
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Union
from torch.utils.data import Dataset

from .. import keys
from .funcs import transform
from ..read import (
    BaseGroup,
    SpaceGroup,
    file_paths_to_readers,
)


@dataclass
class MixDataset(Dataset):
    paths: Union[List[str], str]
    mode: str
    out_type: str
    root_dir: str
    probe_size: int
    atom_cutoff: float
    atom_sel: int
    probe_cutoff: float
    probe_sel: int
    dtype: torch.dtype
    predict_ngfs: np.ndarray
    search_depth: int
    lazy_load: bool
    def __post_init__(self):
        assert self.mode != 'predict' or self.predict_ngfs is not None, 'Predict ngfs must be provided in predict mode.'
        self.readers = file_paths_to_readers(
            self.paths,
            self.out_type,
            self.root_dir,
            self.lazy_load,
            self.search_depth
        )
        self.group_keys: List[str] = []
        self.groups: List[Union[SpaceGroup, BaseGroup]] = []
        self.nframes: List[int] = []
        self.nprobes: List[int] = []
        self.length = 0
        for file_path, reader in self.readers.items():
            for group_key, group in zip(reader.group_keys, reader.groups):
                final = f'{file_path}:{group_key}'
                self.group_keys.append(final)
                self.groups.append(group)
                self.nframes.append(group.nframe)
                if self.out_type != 'potential':
                    if self.mode == 'predict':
                        self.length += group.nframe * math.ceil(np.prod(self.predict_ngfs) / self.probe_size)
                        self.nprobes.append(np.prod(self.predict_ngfs))
                    else:
                        self.length += group.nframe * math.ceil(group.nprobe / self.probe_size)
                        self.nprobes.append(group.nprobe)
                else:
                    self.length += group.nframe
        self.base_seed = 0
    
    def get_igroup_iframe_idots(
        self,
        igroup: int,
        iframe: int,
        iprobes: slice = None,
    ):
        group = self.groups[igroup]
        out_dict = group.get_iframe_iprobes(
            iframe,
            iprobes,
            probe_in_ngfs = self.predict_ngfs,
            return_label = self.mode != 'predict'
        )
        herodata = transform(
            out_dict,
            atom_cutoff=self.atom_cutoff,
            atom_sel=self.atom_sel,
            probe_cutoff=self.probe_cutoff,
            probe_sel=self.probe_sel,
            dtype=self.dtype,
        )
        return herodata
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        igroup = 0
        origin_idx = idx
        while True:
            nframe = self.nframes[igroup]
            ndot = np.prod(self.predict_ngfs) if self.mode == 'predict' else self.nprobes[igroup]
            nbatch = math.ceil(ndot / self.probe_size)
            if idx >= nframe * nbatch:
                idx -= nframe * nbatch
                igroup += 1
                assert igroup < len(self.groups), 'Index out of range.'
            else:
                iframe = idx // nbatch
                ibatch = idx % nbatch
                if self.mode == 'train': # shuffle probe for training
                    seed = origin_idx + self.base_seed
                    rng = np.random.default_rng(seed=seed)
                    idots = rng.choice(ndot, size=self.probe_size, replace=False)
                else:
                    idots = np.arange(ibatch * self.probe_size, min(ndot, (ibatch + 1) * self.probe_size))
                break
        if self.lazy_load and self.mode == 'train':
            idots = np.sort(idots)
        out_data = self.get_igroup_iframe_idots(igroup, iframe, idots)
        out_data[keys.INFOS] = self.groups[igroup].extro_infos
        out_data[keys.FRAME_ID] = f'{self.group_keys[igroup]}:{iframe}'
        return out_data

