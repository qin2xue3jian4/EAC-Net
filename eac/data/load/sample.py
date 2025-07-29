import numpy as np
from dataclasses import dataclass
from ..dataset import MixDataset

MAX_SIZE = 1E6

@dataclass
class EACSampler:
    dataset: MixDataset
    probe_size: int
    frame_shuffle: bool
    probe_shuffle: bool
    base_seed: int = 0
    rank: int = 0
    world_size: int = 1
    def __post_init__(self):
        self.niter = 0
        self.set_probe_size(self.probe_size)
        self.init_env()
    
    def init_env(self):
        ndata = len(self.dataset)
        if self.frame_shuffle:
            ss  = np.random.SeedSequence(self.base_seed, spawn_key=(self.niter))
            rng = np.random.default_rng(ss)
            if ndata > MAX_SIZE:
                full_perm = rng.choice(ndata, size=int(MAX_SIZE), replace=False).tolist()
            else:
                full_perm = rng.permutation(ndata).tolist()
            self.permutation = full_perm[self.rank::self.world_size]
        else:
            self.permutation = np.arange(self.rank, ndata, self.world_size)
        self.iter_length = len(self.permutation)
        self.index = 0
    
    def set_probe_size(self, probe_size: int):
        if self.dataset.out_probe:
            self.probe_size = probe_size
            self.batch_probes = np.ceil(self.dataset.nprobes / probe_size).astype(int)
            # flatten
            self.group_sizes = [f * p for f, p in zip(self.dataset.nframes, self.batch_probes)]
            self.cum_sizes = np.concatenate([[0], np.cumsum(self.group_sizes)])
        else:
            self.cum_sizes = np.concatenate([[0], np.cumsum(self.dataset.nframes)])
        self.dataset.length = int(self.cum_sizes[-1])
    
    def __iter__(self):
        return self
    
    def _idx_to_data_idx(self, idx: int):
        igroup = int(np.searchsorted(self.cum_sizes, idx, side='right') - 1)
        offset = idx - self.cum_sizes[igroup]
        if not self.dataset.out_probe:
            return igroup, offset, None
        group_batch_nprobes = self.batch_probes[igroup]
        iframe = offset // group_batch_nprobes
        ibatch = offset % group_batch_nprobes
        nprobe = self.dataset.nprobes[igroup]
        if self.probe_shuffle: # shuffle probe for training or testing
            ss  = np.random.SeedSequence(self.base_seed, spawn_key=(self.niter, idx))
            rng = np.random.default_rng(ss)
            idots = rng.choice(nprobe, size=self.probe_size, replace=False)
        else:
            idots = np.arange(ibatch * self.probe_size, min(nprobe, (ibatch + 1) * self.probe_size))
        if self.dataset.lazy_load and self.probe_shuffle:
            idots = np.sort(idots)
        return igroup, iframe, idots

    def __next__(self):
        if self.index >= self.iter_length:
            self.niter += 1
            self.init_env()
        idx = self.permutation[self.index]
        self.index += 1
        return self._idx_to_data_idx(idx)