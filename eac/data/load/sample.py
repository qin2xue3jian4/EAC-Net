import numpy as np
from ..dataset import MixDataset

MAX_SIZE = 1E6

class EACSampler:
    def __init__(
        self,
        dataset: MixDataset,
        shuffle: bool,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.ndata = len(dataset)
        self.rank = rank
        self.world_size = world_size
        self.niter = 0
        self.init_env()
    
    def init_env(self):
        if self.shuffle:
            rng = np.random.default_rng(seed=self.niter)
            if self.ndata > MAX_SIZE:
                full_perm = rng.choice(self.ndata, size=int(MAX_SIZE), replace=False).tolist()
            else:
                full_perm = rng.permutation(self.ndata).tolist()
            self.permutation = full_perm[self.rank::self.world_size]
        else:
            self.permutation = np.arange(self.rank, self.ndata, self.world_size)
        self.iter_length = len(self.permutation)
        self.index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= self.iter_length:
            self.niter += 1
            self.init_env()
        idx = self.permutation[self.index]
        self.index += 1
        return idx, self.niter