import os
import tqdm
import torch
import numpy as np
from typing import Dict, List
from collections import defaultdict
from torch import Tensor

from .run import Controller
from ..data import keys, write

class Predictor(Controller):
    def __post_init__(self):
        super().__post_init__()
        frame_size = 1 if self.out_type == 'probe' or self.args.frame_size is None else self.args.frame_size
        probe_size = self.args.probe_size or 50
        epoch_size = -1
        num_workers = self.args.num_workers or 0
        self.loader = self._load_paths_data(
            self.args.paths,
            frame_size=frame_size,
            probe_size=probe_size,
            num_workers=num_workers,
            epoch_size=epoch_size,
        )
        self.model.eval()
    
    def run_probe(self):
        # prepare
        ii, pp = defaultdict(list), defaultdict(list)
        space_keys = [keys.CHARGE]
        if self.spin == 2:
            space_keys.append(keys.CHARGE_DIFF)
        last_frame_id, atom_representations, atom_contributions = 'none', None, None
        iframe = 0
        grid_ptr = 0
        
        for data in tqdm.tqdm(self.loader):
            natom = data[keys.ATOM][keys.POS].shape[0]
            nprobe = data[keys.PROBE][keys.POS].shape[0]
            ngfs = data[keys.PROBE_GRID_NGFS][0].cpu().detach().cpu().numpy().astype(int)
            
            # atom representation
            keep_same_frame = data[keys.FRAME_ID][0] == last_frame_id
            last_frame_id = data[keys.FRAME_ID][0]
            if keep_same_frame and atom_representations is not None:
                data[keys.ATOM][keys.FEATURES] = atom_representations
            if not keep_same_frame:
                iframe += 1
                self._log(f'Predicting frame {iframe}: {last_frame_id}')
            # atom contributions
            if atom_contributions is None and self.args.contribute:
                atom_contributions = torch.zeros((natom*np.prod(ngfs), self.spin), dtype=self.dtype)

            # empty probe
            probe_empty = data[keys.PROBE_EDGE_KEY][keys.INDEX].numel() == 0
            if probe_empty:
                preds = {}
                for key in space_keys:
                    preds[key] = torch.zeros((nprobe,), dtype=self.dtype, device=self.device)
            else:
                preds = self.model(data, out_type=self.out_type, return_atom_features=True)
                if not keep_same_frame:
                    atom_representations = preds[keys.ATOM_FEATURES]
            
            # record preds and grid position
            for key in preds:
                if key not in keys.LABELS:
                    continue
                pp[key].append(preds[key].detach())
            ii[keys.PROBE].append(data[keys.PROBE][keys.POS])
            
            # record node contribution
            if self.args.contribute and not probe_empty:
                edge = data[keys.PROBE_EDGE_KEY]
                atom_index, probe_index = edge[keys.INDEX]
                grid_edge_scalar = edge[keys.CHARGE].detach().cpu()
                linear_indices = (atom_index * np.prod(ngfs) + probe_index + grid_ptr).cpu()
                expanded_indices = linear_indices.unsqueeze(-1).expand(-1, self.spin)
                flat_charge = torch.zeros_like(atom_contributions)
                flat_charge.scatter_add_(0, expanded_indices, grid_edge_scalar)
                atom_contributions += flat_charge
            
            grid_ptr += nprobe
            # output
            if grid_ptr >= np.prod(ngfs):
                preds = {
                    key: torch.cat(value_list, dim=0)
                    for key, value_list in pp.items()
                }
                self.save(data, preds, ngfs, iframe=iframe, value_type='global')
                if self.args.contribute:
                    atom_contributions = atom_contributions.view(natom, np.prod(ngfs), self.spin)
                    for inode, node_value in enumerate(atom_contributions):
                        atom_preds = {}
                        for ispin in range(self.spin):
                            node_spin_value = node_value[:,ispin]
                            if self.spin == 2:
                                value_type = keys.CHARGE if ispin == 0 else keys.CHARGE_DIFF
                            else:
                                value_type = keys.CHARGE
                            atom_preds[value_type] = node_spin_value
                        self.save(data, atom_preds, ngfs, iframe=iframe, value_type=f'atom_{inode}')
                    atom_contributions = None
                grid_ptr = 0
                ii, pp = defaultdict(list), defaultdict(list)
                
        self._log('Finished!')
    def save(
        self,
        data: Dict[str, Tensor],
        preds: Dict[str, Tensor],
        ngfs: np.ndarray,
        iframe: int,
        value_type: str,
    ):
        file = os.path.join(self.output_dir, f'{iframe}_{value_type}.{self.args.format}')
        scalar_data = {
            key: value.reshape(*ngfs).detach().cpu().numpy()
            for key, value in preds.items()
        }
        write.write_data(data, probe_scalars=scalar_data, file_path=file, file_format=self.args.format)
        return None
