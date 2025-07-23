import numpy as np
from typing import Dict
from dataclasses import dataclass

from .. import keys

ATOM_INPUTS = [
    keys.CELL,
    keys.ATOM_POS,
    keys.ATOM_TYPE,
]
ATOM_LABELS = [
    keys.ENERGY,
    keys.FORCE,
    keys.VIRIAL,
]
PROBE_LABELS = [
    keys.CHARGE,
    keys.CHARGE_DIFF,
]

@dataclass
class BaseGroup:
    group: Dict[str, np.ndarray]
    extro_infos: Dict[str, str] = None
    def __post_init__(self):
        self.nframe = self.group[keys.CELL].shape[0]

    def get_iframe(
        self,
        iframe: int,
        return_label: bool=True,
    ):
        out_dict: Dict[str, np.ndarray] = {}
        for key in ATOM_INPUTS:
            out_dict[key] = self.group[key][iframe]
        if return_label:
            for key in ATOM_LABELS:
                if key in self.group:
                    out_dict[key] = np.atleast_1d(self.group[key][iframe])
        return out_dict

class SpaceGroup(BaseGroup):
    def __post_init__(self):
        super().__post_init__()
        if keys.PROBE_GRID_SHAPE in self.group:
            self.sample_probes = keys.PROBE_POS in self.group
            grid_shape = self.group[keys.PROBE_GRID_SHAPE]
            self.data_ngfs = grid_shape[1:]
            self.nprobe = self.group[keys.PROBE_POS].shape[1] if self.sample_probes else np.prod(self.data_ngfs)
    
    def get_iframe_iprobes(
        self,
        iframe: int,
        iprobes: slice=None,
        probe_in_ngfs: np.ndarray=None,
        return_label: bool=True,
    ):
        out_dict = super().get_iframe(iframe, return_label)
        if probe_in_ngfs is None and self.sample_probes:
            out_dict[keys.PROBE_POS] = self.group[keys.PROBE_POS][iframe, iprobes]
        else:
            out_dict[keys.PROBE_POS] = self._get_probe_pos(iframe, iprobes, probe_in_ngfs)
        out_dict[keys.PROBE_GRID_NGFS] = self.data_ngfs if probe_in_ngfs is None else probe_in_ngfs
        if return_label:
            for key in PROBE_LABELS:
                if key in self.group:
                    out_dict[key] = self.group[key][iframe, iprobes]
        return out_dict
    
    def _get_probe_pos(
        self,
        iframe: int,
        igrids: np.ndarray,
        probe_in_ngfs: np.ndarray=None
    ):
        ngfs = self.data_ngfs if probe_in_ngfs is None else probe_in_ngfs
        iprobes = np.unravel_index(igrids, ngfs)
        cell = self.group[keys.CELL][iframe]
        percell = (cell.T / ngfs).T
        probe_pos = np.dot(np.array(iprobes).T, percell)
        return probe_pos
