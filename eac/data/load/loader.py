import logging
import torch
from typing import Dict
from dataclasses import dataclass
from torch.utils.data import DataLoader

from .sample import EACSampler
from ..dataset import MixDataset
from .collate import BatchCollater
from ...utils.envs import setup_seed

@dataclass
class LoaderWrapper:
    dataset: MixDataset
    frame_size: int
    num_workers: int
    shuffle: bool
    epoch_size: int
    device: torch.device
    local_rank: int
    world_size: int
    def __post_init__(self):
        
        self.sampler = EACSampler(
            self.dataset,
            self.shuffle,
            rank=self.local_rank,
            world_size=self.world_size,
        )
        
        if self.epoch_size <= 0:
            self.epoch_size = len(self.sampler.permutation)
        
        self.loader = DataLoader(
            self.dataset,
            self.frame_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            collate_fn=BatchCollater(self.dataset),
            worker_init_fn = setup_seed,
            pin_memory = True,
            prefetch_factor = (3 if self.num_workers > 0 else None),
            persistent_workers = (True if self.num_workers > 0 else False),
        )
        
        self.iteration = iter(self.loader)
        self.idx_ptr = 0

    def load_state_dict(self, state_dict: dict):
        self.sampler.niter = state_dict['niter']
        self.sampler.init_env()
        self.sampler.index = state_dict['index']
        
    def state_dict(self):
        state_dict = {
            'index': self.sampler.index,
            'niter': self.sampler.niter,
        }
        return state_dict
    
        
    def __len__(self):
        return self.epoch_size
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.idx_ptr == self.epoch_size:
            self.idx_ptr = 0
            raise StopIteration
        try:
            outdata = next(self.iteration)
        except StopIteration:
            self.iteration = iter(self.loader)
            outdata = next(self.iteration)
        outdata2: Dict[str, Dict[str, torch.Tensor]] = {}
        for key, value in outdata.items():
            if isinstance(value, dict):
                outdata2[key] = {
                    subkey: subvalue.to(self.device) if isinstance(subvalue, torch.Tensor) else subvalue
                    for subkey, subvalue in value.items()
                }
            else:
                outdata2[key] = value.to(self.device) if isinstance(value, torch.Tensor) else value

        self.idx_ptr += 1
        return outdata2

    def close(self):
        for reader in self.dataset.readers.values():
            reader.close()
        if self.loader.persistent_workers:
            try:
                self.loader._iterator._shutdown_workers()
            except Exception as e:
                logging.error('Error when closing loader.')
        del self.iteration
        del self.loader
        return