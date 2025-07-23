import torch
from torch_geometric.loader.dataloader import Collater

from .. import keys

class BatchCollater(Collater):
    def __init__(self, dataset):
        super().__init__(dataset, [keys.ATOM, keys.PROBE])
    def __call__(self, batch):
        data_or_batch = super().__call__(batch)
        for key, value in data_or_batch._global_store.items():
            if isinstance(value, torch.Tensor) and value.ndim == batch[0][key].ndim:
                data_or_batch[key] = value.view(len(batch), -1, *value.shape[1:])
        dict_data = {
            keys.ATOM: {
                keys.POS: data_or_batch[keys.ATOM][keys.POS],
                keys.TYPE: data_or_batch[keys.ATOM][keys.TYPE],
                keys.BATCH: data_or_batch[keys.ATOM][keys.BATCH],
                keys.NUM_NODES: data_or_batch[keys.ATOM].num_nodes,
            },
            keys.PROBE: {
                keys.BATCH: data_or_batch[keys.PROBE][keys.BATCH],
                keys.NUM_NODES: data_or_batch[keys.PROBE].num_nodes,
                keys.POS: data_or_batch[keys.PROBE][keys.POS],
            },
            keys.PROBE_EDGE_KEY: {
                keys.INDEX: data_or_batch[keys.PROBE_EDGE_KEY][keys.INDEX],
                keys.PERIODIC_INDEX: data_or_batch[keys.PROBE_EDGE_KEY][keys.PERIODIC_INDEX],
            },
            keys.ATOM_EDGE_KEY: {
                keys.INDEX: data_or_batch[keys.ATOM_EDGE_KEY][keys.INDEX],
                keys.PERIODIC_INDEX: data_or_batch[keys.ATOM_EDGE_KEY][keys.PERIODIC_INDEX],
            },
            keys.GLOBAL: {
                keys.CELL: data_or_batch[keys.CELL],
            },
            keys.FRAME_ID: data_or_batch[keys.FRAME_ID],
            keys.INFOS: data_or_batch[keys.INFOS],
            keys.PROBE_GRID_NGFS: data_or_batch[keys.PROBE_GRID_NGFS],
        }
        
        for label in keys.LABELS.values():
            parent = data_or_batch[label.parent] if label.parent != keys.GLOBAL else data_or_batch
            if keys.REAL_PREFIX+label.key in parent:
                dict_data[label.parent][keys.REAL_PREFIX+label.key] = parent[keys.REAL_PREFIX+label.key]
        return dict_data
