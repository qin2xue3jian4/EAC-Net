import math
import torch
import numpy as np
from collections import defaultdict
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
    out_probe: bool
    root_dir: str
    atom_cutoff: float
    atom_sel: int
    probe_cutoff: float
    probe_sel: int
    dtype: torch.dtype
    ngfs_str: str
    search_depth: int
    lazy_load: bool
    exclude_keys: List[str]
    def __post_init__(self):
        assert self.mode != 'predict' or self.ngfs_str is not None, 'Predict ngfs must be provided in predict mode.'
        self.readers = file_paths_to_readers(
            self.paths,
            self.out_probe,
            self.root_dir,
            self.lazy_load,
            self.search_depth
        )
        self.ngroups = [len(reader.groups) for reader in self.readers.values()]
        self.group_keys: List[str] = []
        self.groups: List[Union[SpaceGroup, BaseGroup]] = []
        self.nframes: List[int] = []
        self.has_diff = True
        for file_path, reader in self.readers.items():
            for group_key, group in zip(reader.group_keys, reader.groups):
                if self.exclude_keys is not None and group_key in self.exclude_keys:
                    continue
                if keys.CHARGE_DIFF not in group.group:
                    self.has_diff = False
                final = f'{file_path}|{group_key}'
                self.group_keys.append(final)
                self.groups.append(group)
                self.nframes.append(group.nframe)
        if self.out_probe and self.mode == 'predict':
            self.predict_ngfs = self.get_predict_ngfs()
        if self.out_probe:
            self.nprobes = self.get_nprobes()
        self.length = len(self.groups)
    
    def get_nprobes(self):
        nprobes = []
        for igroup, group in enumerate(self.groups):
            if hasattr(self, 'predict_ngfs'):
                nprobe = np.prod(self.predict_ngfs[igroup])
            else:
                nprobe = group.nprobe
            nprobes.append(nprobe)
        return np.array(nprobes)
    
    def get_predict_ngfs(self):
        predict_ngfs = []
        for group in self.groups:
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
                    raise ValueError(f'Error {e} ngfs input: {self.ngfs_str}.')
            predict_ngfs.append(ngfs)
        return predict_ngfs
    
    def get_igroup_iframe_idots(
        self,
        igroup: int,
        iframe: int,
        iprobes: slice = None,
    ):
        group = self.groups[igroup]
        if self.out_probe:
            ngfs = self.predict_ngfs[igroup] if self.mode == 'predict' else None
            out_dict = group.get_iframe_iprobes(
                iframe,
                iprobes,
                probe_in_ngfs = ngfs,
                return_label = self.mode != 'predict',
                return_diff = self.has_diff
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
        if self.mode != 'train':
            herodata[keys.GLOBAL][keys.INFOS] = self.groups[igroup].extro_infos
        herodata[keys.GLOBAL][keys.FRAME_ID] = f'{self.group_keys[igroup]}|{iframe}'
        return herodata
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            igroup, iframe, idots = idx
            return self.get_igroup_iframe_idots(igroup, iframe, idots)
        return self.groups[idx]
    
    def collect_infos(self):
        infos = {}
        for label_name, label in keys.LABELS.items():
            if self.out_probe and not label.probe:
                continue
            num = 0
            value_sum, value_2sum = 0.0, 0.0
            for group in self.groups:
                if label.key not in group.group:
                    continue
                num += group.group[label.key].size
                value_sum += group.group[label.key].sum()
                value_2sum += np.square(group.group[label.key]).sum()
            if num == 0:
                continue
            infos[f'{label.key}_mean'] = float(value_sum / num)
            infos[f'{label.key}_std'] = float(math.sqrt(value_2sum / num - math.pow(value_sum / num, 2)))
        return infos
    
            