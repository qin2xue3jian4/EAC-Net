import os
import torch
from typing import Dict

def get_periodic_indexs():
    nx = int(os.environ.get('nx', '1'))
    ny = int(os.environ.get('ny', '1'))
    nz = int(os.environ.get('nz', '1'))
    ranges = [torch.arange(-n, n+1) for n in [nx, ny, nz]]
    offsets = torch.cartesian_prod(*ranges)
    __, order = torch.sort(torch.sum(offsets**2, dim=1), stable=True)
    offsets = offsets[order]
    return offsets

ELECTRONIC_LAYOUTS = {
    'H': [1, 0, 0, 0, 0, 0, 0],
    'He': [2, 0, 0, 0, 0, 0, 0],
    'Li': [2, 1, 0, 0, 0, 0, 0],
    'Be': [2, 2, 0, 0, 0, 0, 0],
    'B': [2, 3, 0, 0, 0, 0, 0],
    'C': [2, 4, 0, 0, 0, 0, 0],
    'N': [2, 5, 0, 0, 0, 0, 0],
    'O': [2, 6, 0, 0, 0, 0, 0],
    'F': [2, 7, 0, 0, 0, 0, 0],
    'Ne': [2, 8, 0, 0, 0, 0, 0],
    'Na': [2, 8, 1, 0, 0, 0, 0],
    'Mg': [2, 8, 2, 0, 0, 0, 0],
    'Al': [2, 8, 3, 0, 0, 0, 0],
    'Si': [2, 8, 4, 0, 0, 0, 0],
    'P': [2, 8, 5, 0, 0, 0, 0],
    'S': [2, 8, 6, 0, 0, 0, 0],
    'Cl': [2, 8, 7, 0, 0, 0, 0],
    'Ar': [2, 8, 8, 0, 0, 0, 0],
    'K': [2, 8, 8, 1, 0, 0, 0],
    'Ca': [2, 8, 8, 2, 0, 0, 0],
    'Sc': [2, 8, 9, 2, 0, 0, 0],
    'Ti': [2, 8, 10, 2, 0, 0, 0],
    'V': [2, 8, 11, 2, 0, 0, 0],
    'Cr': [2, 8, 13, 1, 0, 0, 0],
    'Mn': [2, 8, 13, 2, 0, 0, 0],
    'Fe': [2, 8, 14, 2, 0, 0, 0],
    'Co': [2, 8, 15, 2, 0, 0, 0],
    'Ni': [2, 8, 16, 2, 0, 0, 0],
    'Cu': [2, 8, 18, 1, 0, 0, 0],
    'Zn': [2, 8, 18, 2, 0, 0, 0],
    'Ga': [2, 8, 18, 3, 0, 0, 0],
    'Ge': [2, 8, 18, 4, 0, 0, 0],
    'As': [2, 8, 18, 5, 0, 0, 0],
    'Se': [2, 8, 18, 6, 0, 0, 0],
    'Br': [2, 8, 18, 7, 0, 0, 0],
    'Kr': [2, 8, 18, 8, 0, 0, 0],
    'Rb': [2, 8, 18, 8, 1, 0, 0],
    'Sr': [2, 8, 18, 8, 2, 0, 0],
    'Y': [2, 8, 18, 9, 2, 0, 0],
    'Zr': [2, 8, 18, 10, 2, 0, 0],
    'Nb': [2, 8, 18, 12, 1, 0, 0],
    'Mo': [2, 8, 18, 13, 1, 0, 0],
    'Tc': [2, 8, 18, 13, 2, 0, 0],
    'Ru': [2, 8, 18, 15, 1, 0, 0],
    'Rh': [2, 8, 18, 16, 1, 0, 0],
    'Pd': [2, 8, 18, 18, 0, 0, 0],
    'Ag': [2, 8, 18, 18, 1, 0, 0],
    'Cd': [2, 8, 18, 18, 2, 0, 0],
    'In': [2, 8, 18, 18, 3, 0, 0],
    'Sn': [2, 8, 18, 18, 4, 0, 0],
    'Sb': [2, 8, 18, 18, 5, 0, 0],
    'Te': [2, 8, 18, 18, 6, 0, 0],
    'I': [2, 8, 18, 18, 7, 0, 0],
    'Xe': [2, 8, 18, 18, 8, 0, 0],
    'Cs': [2, 8, 18, 18, 8, 1, 0],
    'Ba': [2, 8, 18, 18, 8, 2, 0],
    'La': [2, 8, 18, 18, 9, 2, 0],
    'Ce': [2, 8, 18, 19, 9, 2, 0],
    'Pr': [2, 8, 18, 21, 8, 2, 0],
    'Nd': [2, 8, 18, 22, 8, 2, 0],
    'Pm': [2, 8, 18, 23, 8, 2, 0],
    'Sm': [2, 8, 18, 24, 8, 2, 0],
    'Eu': [2, 8, 18, 25, 8, 2, 0],
    'Gd': [2, 8, 18, 25, 9, 2, 0],
    'Tb': [2, 8, 18, 27, 8, 2, 0],
    'Dy': [2, 8, 18, 28, 8, 2, 0],
    'Ho': [2, 8, 18, 29, 8, 2, 0],
    'Er': [2, 8, 18, 30, 8, 2, 0],
    'Tm': [2, 8, 18, 31, 8, 2, 0],
    'Yb': [2, 8, 18, 32, 8, 2, 0],
    'Lu': [2, 8, 18, 32, 9, 2, 0],
    'Hf': [2, 8, 18, 32, 10, 2, 0],
    'Ta': [2, 8, 18, 32, 11, 2, 0],
    'W': [2, 8, 18, 32, 12, 2, 0],
    'Re': [2, 8, 18, 32, 13, 2, 0],
    'Os': [2, 8, 18, 32, 14, 2, 0],
    'Ir': [2, 8, 18, 32, 15, 2, 0],
    'Pt': [2, 8, 18, 32, 17, 1, 0],
    'Au': [2, 8, 18, 32, 18, 1, 0],
    'Hg': [2, 8, 18, 32, 18, 2, 0],
    'Tl': [2, 8, 18, 32, 18, 3, 0],
    'Pb': [2, 8, 18, 32, 18, 4, 0],
    'Bi': [2, 8, 18, 32, 18, 5, 0],
    'Po': [2, 8, 18, 32, 18, 6, 0],
    'At': [2, 8, 18, 32, 18, 7, 0],
    'Rn': [2, 8, 18, 32, 18, 8, 0],
    'Fr': [2, 8, 18, 32, 18, 8, 1],
    'Ra': [2, 8, 18, 32, 18, 8, 2],
    'Ac': [2, 8, 18, 32, 18, 9, 2],
    'Th': [2, 8, 18, 32, 18, 10, 2],
    'Pa': [2, 8, 18, 32, 20, 9, 2],
    'U': [2, 8, 18, 32, 21, 9, 2],
    'Np': [2, 8, 18, 32, 22, 9, 2],
    'Pu': [2, 8, 18, 32, 24, 8, 2],
    'Am': [2, 8, 18, 32, 25, 8, 2],
    'Cm': [2, 8, 18, 32, 25, 9, 2],
    'Bk': [2, 8, 18, 32, 27, 8, 2],
    'Cf': [2, 8, 18, 32, 28, 8, 2],
    'Es': [2, 8, 18, 32, 29, 8, 2],
    'Fm': [2, 8, 18, 32, 30, 8, 2],
    'Md': [2, 8, 18, 32, 31, 8, 2],
    'No': [2, 8, 18, 32, 32, 8, 2],
    'Lr': [2, 8, 18, 32, 32, 8, 3],
}
ELEMENTS = list(ELECTRONIC_LAYOUTS.keys())
BOHR = 1.8897259886
PERIODIC_INDEXS = get_periodic_indexs()

def get_physics_encode(
    physics_args: Dict,
    start_in_one: bool = True,
):
    physics_encode = []
    
    if physics_args.get('electronic_layout', 7) > 0:
        max_layout = physics_args.get('electronic_layout', 7)
        max_layout_nums = torch.arange(1, max_layout+1) ** 2 * 2
        electronic_layout_normalize = torch.tensor(list(ELECTRONIC_LAYOUTS.values()))[:,:max_layout] / max_layout_nums
        physics_encode.append(electronic_layout_normalize)
    
    if physics_args.get('atomic_number', False):
        physics_encode.append(torch.arange(1,len(ELEMENTS)+1)[:,None])
    
    physics_encode = torch.concatenate(physics_encode, dim=-1)
    
    if start_in_one:
        zero_row = torch.zeros(1, physics_encode.shape[1])
        physics_encode = torch.cat([zero_row, physics_encode], dim=0)
    
    return physics_encode

