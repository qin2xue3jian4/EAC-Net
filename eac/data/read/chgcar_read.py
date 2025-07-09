import numpy as np
from typing import ClassVar, List
from ase.calculators.vasp import (
    VaspChargeDensity
)

from .. import keys
from .base import BaseReader, ReaderFactory

def chgcar_to_group(chgcar_path: str):
    vcd = VaspChargeDensity(chgcar_path)
    nframe = len(vcd.atoms)
    group = {
        keys.PROBE_GRID_SHAPE: np.array([nframe, *(vcd.chg[0].shape)]),
        keys.FILE_SOURCE: chgcar_path,
    }
    group[keys.ATOM_TYPE] = np.vstack([atom.numbers for atom in vcd.atoms]).reshape(nframe, -1)
    group[keys.ATOM_POS] = np.vstack([atom.positions for atom in vcd.atoms]).reshape(nframe, -1, 3)
    group[keys.CELL] = np.vstack([atom.cell.array for atom in vcd.atoms]).reshape(nframe, 3, 3)
    group[keys.CHARGE] = np.vstack([c[None] for c in vcd.chg]).reshape(nframe, -1)
    if hasattr(vcd, 'chgdiff') and len(getattr(vcd, 'chgdiff')) > 0:
        group[keys.CHARGE_DIFF] = np.vstack([c[None] for c in vcd.chgdiff]).reshape(nframe, -1)
    
    extro_infos = {}
    if hasattr(vcd, 'aug'):
        extro_infos[keys.CHGCAR_AUG] = vcd.aug
    if hasattr(vcd, 'augdiff'):
        extro_infos[keys.CHGCAR_AUG_DIFF] = vcd.augdiff
    
    return group, extro_infos

@ReaderFactory.register('chg')
class VASPReader(BaseReader):
    
    patterns: ClassVar[List[str]] = ['*.chgcar', '*.CHGCAR']
    pattern_parent: ClassVar[bool] = False
    only_structure: ClassVar[bool] = False
    
    def load_file(self):
        group, extro_infos = chgcar_to_group(self.file_path)
        return {'chgcar': (group, extro_infos)}
    