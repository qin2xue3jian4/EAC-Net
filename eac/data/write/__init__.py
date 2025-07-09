import torch
import numpy as np
from typing import Dict
from ase.calculators.vasp import VaspChargeDensity

from ase import Atoms
from .. import keys

def write_data(
    data: Dict[str, Dict[str, torch.Tensor]],
    file_path: str,
    probe_scalars: Dict[str, np.ndarray] = None,
    file_format: str = 'chgcar',
):
    atom_pos = data[keys.ATOM][keys.POS].detach().cpu().numpy()
    cell = data[keys.GLOBAL][keys.CELL][0].detach().cpu().numpy()
    atom_type = data[keys.ATOM][keys.TYPE].detach().cpu().numpy()
    crystal = Atoms(
        numbers=atom_type,
        positions=atom_pos,
        cell=cell,
        pbc=True,
    )
    if probe_scalars is not None:
        chgcar = VaspChargeDensity(filename=None)
        chgcar.atoms.append(crystal)
        for key in ['aug', 'augdiff']:
            if key in data['infos']:
                string = data['infos'][key][0]
                setattr(chgcar, key, string)
        for key, value in probe_scalars.items():
            values = getattr(chgcar, key)
            values.append(value)
    if file_format == 'chgcar':
        chgcar.write(filename=file_path, format='chgcar')
    return
