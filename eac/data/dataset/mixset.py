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
    ngfs_str: str
    search_depth: int
    lazy_load: bool
    base_seed: int = 0
    def __post_init__(self):
        assert self.mode != 'predict' or self.ngfs_str is not None, 'Predict ngfs must be provided in predict mode.'
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
        self.batch_probes: List[int] = []
        self.predict_ngfs = []
        for file_path, reader in self.readers.items():
            for group_key, group in zip(reader.group_keys, reader.groups):
                final = f'{file_path}:{group_key}'
                self.group_keys.append(final)
                self.groups.append(group)
                self.nframes.append(group.nframe)
                if self.out_type != 'potential':
                    if self.mode == 'predict':
                        if '*' in self.ngfs_str:
                            ngfs = np.array(self.ngfs_str.split('*'), dtype=np.int64)
                        elif self.ngfs_str == 'origin':
                            assert hasattr(group, 'data_ngfs'), 'Origin ngfs mode need data contains ngfs info.'
                            ngfs = group.data_ngfs
                        else:
                            try:
                                length = float(self.ngfs_str)
                                cell_lengths = np.linalg.norm(group.group['cell'], axis=-1).min(axis=0)
                                ngfs = np.ceil(cell_lengths / length).astype(int)
                            except ValueError as e:
                                raise ValueError('Error ngfs input.')
                        self.predict_ngfs.append(ngfs)
                        nprobe = np.prod(ngfs)
                    else:
                        nprobe = group.nprobe
                    n_batch_probes = math.ceil(nprobe / self.probe_size)
                else:
                    n_batch_probes = nprobe = 1
                self.nprobes.append(nprobe)
                self.batch_probes.append(n_batch_probes)
        # flatten
        self.group_sizes = [f * p for f, p in zip(self.nframes, self.batch_probes)]
        self.cum_sizes = np.concatenate([[0], np.cumsum(self.group_sizes)])
        self.length = int(self.cum_sizes[-1])
        
    def get_igroup_iframe_idots(
        self,
        igroup: int,
        iframe: int,
        iprobes: slice = None,
    ):
        group = self.groups[igroup]
        if isinstance(group, SpaceGroup):
            ngfs = self.predict_ngfs[igroup] if self.mode == 'predict' else None
            out_dict = group.get_iframe_iprobes(
                iframe,
                iprobes,
                probe_in_ngfs = ngfs,
                return_label = self.mode != 'predict'
            )
        else:
            out_dict = group.get_iframe(
                iframe,
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
        herodata[keys.INFOS] = self.groups[igroup].extro_infos
        herodata[keys.FRAME_ID] = f'{self.group_keys[igroup]}:{iframe}'
        return herodata
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, niter = idx
        else:
            niter = 0
        igroup = int(np.searchsorted(self.cum_sizes, idx, side='right') - 1)
        offset = idx - self.cum_sizes[igroup]
        group_batch_nprobes = self.batch_probes[igroup]
        iframe = offset // group_batch_nprobes
        ibatch = offset % group_batch_nprobes
        nprobe = self.nprobes[igroup]
        if self.mode == 'train': # shuffle probe for training
            ss  = np.random.SeedSequence(self.base_seed, spawn_key=(niter, idx))
            rng = np.random.default_rng(ss)
            idots = rng.choice(nprobe, size=self.probe_size, replace=False)
        else:
            idots = np.arange(ibatch * self.probe_size, min(nprobe, (ibatch + 1) * self.probe_size))
        if self.lazy_load and self.mode == 'train':
            idots = np.sort(idots)
        return self.get_igroup_iframe_idots(igroup, iframe, idots)
