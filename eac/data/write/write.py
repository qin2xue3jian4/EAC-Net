import h5py
import torch
import numpy as np
from typing import Dict, Union, List
from ase.calculators.vasp import VaspChargeDensity
from ase import Atoms

from .. import keys
from ..read.group import BaseGroup, SpaceGroup

class Writer:
    def __init__(self, save: bool = True):
        self.save = save
        self.groups = []
        self.group_keys = []
    
    def append_result(
        self,
        frame_id: str,
        space_group: SpaceGroup,
        predict_results: List[Dict[str, torch.Tensor]],
        sample_probe: bool,
        ngfs: np.ndarray,
    ):
        npy_group = {}
        for key, value in space_group.group.items():
            if key in [keys.CHARGE, keys.CHARGE_DIFF, keys.PROBE_POS]:
                continue
            npy_group[key] = value
        
        nframe = len(predict_results)
        for key in [keys.CHARGE, keys.CHARGE_DIFF]:
            if f'pred_{key}' in predict_results[0]:
                npy_group[key] = np.vstack([
                    result[f'pred_{key}'].cpu().numpy().flatten()
                    for result in predict_results
                ]).reshape(nframe, -1)
        
        npy_group[keys.PROBE_GRID_SHAPE] = np.array([len(predict_results), *ngfs])
        
        if sample_probe:
            npy_group[keys.PROBE_POS] = np.vstack([
                result[keys.PROBE_POS].cpu().numpy().flatten()
                for result in predict_results
            ]).reshape(nframe, -1, 3)
        
        new_group = SpaceGroup(
            npy_group,
            space_group.extro_infos,
        )
        self.groups.append(new_group)
        group_key = frame_id.split('|')[1]
        self.group_keys.append(group_key)
    
    def write_to_h5(self, h5_file: str):
        with h5py.File(h5_file, 'w') as new_h5:
            for group_key, space_group in zip(self.group_keys, self.groups):
                space_group: Union[BaseGroup, SpaceGroup]
                new_h5.create_group(group_key)
                for item_key, item_value in space_group.group.items():
                    if isinstance(item_value, np.ndarray):
                        new_h5[group_key].create_dataset(item_key, data=item_value)
                    else:
                        new_h5[group_key].attrs[item_key] = item_value
                for attr_key, attr_value in space_group.extro_infos.items():
                    new_h5[group_key].attrs[attr_key] = attr_value
        return
    
    def write_to_chgcar(
        self,
        chgcar_file: str,
        grid_scalars: Dict=None,
        iframe: int=None,
        ngfs: np.ndarray=None,
    ):
        assert len(self.groups) == 1, "CHGCAR can only write one group"
        group: SpaceGroup = self.groups[0]
        assert keys.PROBE_POS not in group.group, "CHGCAR can not write sample data."
        
        vcd = VaspChargeDensity(filename=None)
        
        if iframe is None:
            iframes = range(group.group[keys.ATOM_POS].shape[0])
        else:
            iframes = [iframe]
        for iframe in iframes:
            atom_pos = group.group[keys.ATOM_POS][iframe]
            cell = group.group[keys.CELL][iframe]
            atom_type = group.group[keys.ATOM_TYPE][iframe]
            crystal = Atoms(
                numbers=atom_type,
                positions=atom_pos,
                cell=cell,
                pbc=True,
            )
            vcd.atoms.append(crystal)
        
        for key in ['aug', 'augdiff']:
            if key in group.extro_infos:
                string = group.extro_infos[key]
                setattr(vcd, key, string)
        
        if grid_scalars is None: grid_scalars = group.group
        if ngfs is None: ngfs = group.group[keys.PROBE_GRID_SHAPE][1:]
        for key in [keys.CHARGE, keys.CHARGE_DIFF]:
            if key not in grid_scalars:
                continue
            values = getattr(vcd, key)
            reshaped = [flattened_chg.reshape(*ngfs) for flattened_chg in grid_scalars[key]]
            values.extend(reshaped)
        vcd.write(filename=chgcar_file, format='chgcar')

    def write_to_file(self, file: str):
        if not self.save: return
        if file.endswith('.h5'):
            self.write_to_h5(file)
        elif file.endswith('.chgcar'):
            self.write_to_chgcar(file)
        else:
            raise ValueError('file format not supported')
        