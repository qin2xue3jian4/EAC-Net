import torch
from torch_geometric.loader.dataloader import Collater

from .. import keys

class BatchCollater(Collater):
    def __init__(self, dataset):
        super().__init__(dataset, [keys.ATOM, keys.PROBE])
    def __call__(self, batch):
        data_or_batch = super().__call__(batch)
        for key, value in data_or_batch[keys.GLOBAL].items():
            if isinstance(value, torch.Tensor) and value.ndim == batch[0][keys.GLOBAL][key].ndim:
                data_or_batch[keys.GLOBAL][key] = value.view(len(batch), -1, *value.shape[1:])
        return data_or_batch
