import os
import dpdata
import numpy as np
from typing import ClassVar, List, Union

from .. import keys
from ..physics import ELEMENTS
from .chgcar_read import chgcar_to_group
from .base import BaseReader, ReaderFactory


dpdata_keys = {
    'energies': keys.ENERGY,
    'forces': keys.FORCE,
    'virials': keys.VIRIAL,
    'coords': keys.ATOM_POS,
    'cells': keys.CELL,
}

def system_to_group(system: Union[dpdata.LabeledSystem, dpdata.System]):
    group = {
        eacnet_key: system.data[dpdata_key]
        for dpdata_key, eacnet_key in dpdata_keys.items()
        if dpdata_key in system.data
    }
    nframe = group[keys.ATOM_POS].shape[0]
    atom_ids = np.array([ELEMENTS.index(atom)+1 for atom in system.data['atom_names']])
    group[keys.ATOM_TYPE] = np.repeat(atom_ids[system.data['atom_types']][None], nframe, axis=0)
    return group

@ReaderFactory.register('vasp')
class VASPReader(BaseReader):
    
    patterns: ClassVar[List[str]] = ['CHGCAR', 'OUTCAR',]
    pattern_parent: ClassVar[bool] = True
    only_structure: ClassVar[bool] = False
    
    def load_file(self):
        
        # chgcar
        chgcar = os.path.join(self.file_path, 'CHGCAR')
        if os.path.exists(chgcar):
            group, extro_infos = chgcar_to_group(chgcar)
        else:
            group = {
                keys.FILE_SOURCE: self.file_path
            }
            extro_infos = {}
        
        # outcar
        outcar = os.path.join(self.file_path, 'OUTCAR')
        if os.path.exists(outcar):
            system = dpdata.LabeledSystem(outcar, 'vasp/outcar')
            group.update(system_to_group(system))
        
        return {'vasp': (group, extro_infos)}

@ReaderFactory.register('poscar')
class POSCARReader(BaseReader):
    
    patterns: ClassVar[List[str]] = ['POSCAR', '*.POSCAR', '*.poscar', '*.vasp']
    pattern_parent: ClassVar[bool] = False
    only_structure: ClassVar[bool] = True
    
    def load_file(self):
        
        system = dpdata.System(self.file_path, 'vasp/poscar')
        group = system_to_group(system)
        
        return {'poscar': (group, {})}