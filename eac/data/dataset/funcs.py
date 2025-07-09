import torch
import numpy as np
from torch import Tensor
from torch_cluster import radius
from typing import Dict, List, Union
from torch_geometric.data import HeteroData

from .. import keys
from ..physics import get_periodic_indexs

DEFAULT_NODE_SEL = 40
periodic_indexs = get_periodic_indexs().numpy().astype(np.float64)

# neighbor list build
def _edge_src_index_optimized(edge_src):
    """
    Generate group marker (True indicates the start of a new group)
    Calculate the group number to which each element belongs
    Find the starting index for each group
    Calculate the index of each element within the group
    """
    mask = torch.cat([torch.ones(1, dtype=torch.bool, device=edge_src.device),
                      edge_src[1:] != edge_src[:-1]])
    group_ids = mask.cumsum(dim=0) - 1
    group_starts = torch.where(mask)[0]
    indices = torch.arange(len(edge_src), device=edge_src.device) - group_starts[group_ids]
    return indices

def radius_3d(
    c_pos: Tensor,
    n_pos: Tensor,
    periodic_vectors: Tensor,
    rcut: float,
    sel: int,
    skip_self_neighbor: bool,
    strict_nearest: bool = True,
):
    """Search for the adjacency table of (center - neighbor - periodic) vectors.
    Args:
        c_pos: center coord, (nc, ndim)
        n_pos: neighbor coord, (nn, ndim)
        periodic_vectors: To skip self neighbors, it is necessary to ensure that the first vector is (0,0,0), (np, ndim)
        rcut: cut-off radius, float
        sel: maximum number of neighbors, int
        skip_self_neighbor: skip self neighboring, bool
        strict_nearest: whether to strictly return neighbors, bool
    Returns:
        center_index: index of center coord, (nedge)
        neighbor_index: index of neighbor coord, (nedge)
        periodic_index: index of periodic vectors, (nedge)
    """
    periodic_n_pos = n_pos[None,:,:] + periodic_vectors[:,None,:]
    __, nn, ndim = periodic_n_pos.shape
    periodic_n_pos = periodic_n_pos.view(-1, ndim)
    if strict_nearest and c_pos.device.type == 'cpu':
        max_num_neighbors = 5 * sel
    elif strict_nearest:
        max_num_neighbors = 2 * sel
    else:
        max_num_neighbors = sel
    center_index, periodic_n_index = radius(
        periodic_n_pos,
        c_pos,
        rcut,
        max_num_neighbors = max_num_neighbors,
        ignore_same_index = skip_self_neighbor,
    )
    if strict_nearest:
        fake_distances = torch.linalg.norm(
            c_pos[center_index] - periodic_n_pos[periodic_n_index],
            dim=-1,
        ) / rcut / 10 + center_index
        sort_order = torch.argsort(fake_distances)
        periodic_n_index = periodic_n_index[sort_order]
        indices = _edge_src_index_optimized(center_index)
        mask = indices < sel
        periodic_n_index = periodic_n_index[mask]
        center_index = center_index[mask]
    neighbor_index = periodic_n_index % nn
    periodic_index = periodic_n_index // nn
    index_3d = torch.vstack((center_index, neighbor_index, periodic_index))
    return index_3d

def atom_coord_to_index(
    atom_coord: Union[Tensor, np.ndarray],
    atom_numbers: Union[Tensor, np.ndarray],
    periodic_vectors: Union[Tensor, np.ndarray],
    rcut: float,
    sels: Union[int, List[int], None] = None,
    atom_types: Union[Tensor, List] = None,
):
    """Search for the adjacency table between atoms
    Args:
        atom_coord: atom coords, (natom, ndim)
        atom_numbers: atom numbers, (natom)
        periodic_vectors: To skip self neighbors, it is necessary to ensure that the first vector is (0,0,0), (np, ndim)
        rcut: cut-off radius, float
        sels: maximum number of neighbors, if it is list, it should correspond to elements sorted by the periodic table, and its length should be consistent
    Returns:
        index_3d: 中心坐标的索引, (3, nedge), 分别对应中心坐标, 邻域坐标, 周期性矢量
    """

    if sels is None or isinstance(sels, int):
        sel = DEFAULT_NODE_SEL if sels is None else sels
        atom_types = torch.sort(torch.unique(atom_numbers))[0]
        sels = [sel,] * len(atom_types)
    else:
        assert atom_types is not None
        assert len(sels) == len(atom_types)
    
    c_indexs, n_indexs, p_indexs = [], [], []
    for c_type in atom_types:
        c_mask = atom_numbers == c_type
        c_coord = atom_coord[c_mask]
        for sel, n_type in zip(sels, atom_types):
            n_mask = atom_numbers == n_type
            n_coord = atom_coord[n_mask]
            c_index, n_index, p_index = radius_3d(
                c_coord,
                n_coord,
                periodic_vectors,
                rcut,
                sel,
                c_type == n_type,
            )
            rn_index = torch.where(n_mask)[0][n_index]
            rc_index = torch.where(c_mask)[0][c_index]
            c_indexs.append(rc_index)
            n_indexs.append(rn_index)
            p_indexs.append(p_index)
    c = torch.cat(c_indexs)
    n = torch.cat(n_indexs)
    p = torch.cat(p_indexs)
    edge_index_ed = torch.vstack((c, n, p))
    return edge_index_ed

# transform
def npydict_to_tensordict(
    npydict: Dict[str, np.ndarray],
    dtype: torch.dtype,
):
    npydict[keys.PERIODIC_VECTORS] = np.matmul(periodic_indexs, npydict[keys.CELL])
    tensordict = {}
    for key, value in npydict.items():
        if isinstance(value, np.ndarray):
            tensorvalue = torch.from_numpy(value)
            if key == keys.ATOM_TYPE:
                tensordict[key] = tensorvalue.to(dtype=torch.long)
            else:
                tensordict[key] = tensorvalue.to(dtype=dtype)
        else:
            tensordict[key] = value
    return tensordict

def construct_adjacency(
    tensordict: Dict[str, torch.Tensor],
    atom_cutoff: float,
    atom_sel: int,
    probe_cutoff: float,
    probe_sel: int,
):
    periodic_vectors = tensordict[keys.PERIODIC_VECTORS]
    # atom
    atom_pos = tensordict[keys.ATOM_POS]
    atom_numbers = tensordict[keys.ATOM_TYPE]
    atom_index_3d = atom_coord_to_index(
        atom_pos,
        atom_numbers,
        periodic_vectors,
        atom_cutoff,
        atom_sel,
    )
    # probe
    probe_pos = tensordict[keys.PROBE_POS]
    probe_index_3d = radius_3d(
        probe_pos,
        atom_pos,
        periodic_vectors,
        rcut=probe_cutoff,
        sel=probe_sel,
        skip_self_neighbor=False,
        strict_nearest=True,
    )
    tensordict[keys.ATOM_EDGE_INDEX] = atom_index_3d
    tensordict[keys.PROBE_EDGE_INDEX] = probe_index_3d
    return tensordict

def tensordict_to_heterodata(tensordict: Dict[str, torch.Tensor]):
    heterodata = HeteroData()
    # node and global
    heterodata[keys.ATOM].pos = tensordict[f'{keys.ATOM}_{keys.POS}']
    heterodata[keys.ATOM].type = tensordict[f'{keys.ATOM}_{keys.TYPE}']
    heterodata[keys.PROBE].pos = tensordict[f'{keys.PROBE}_{keys.POS}']
    heterodata.cell = tensordict[keys.CELL]
    # edge
    for CENTER_KEY, EDGE_KEY in zip([keys.ATOM, keys.PROBE], [keys.ATOM_EDGE_KEY, keys.PROBE_EDGE_KEY]):
        heterodata[EDGE_KEY][keys.INDEX] = tensordict[f'{CENTER_KEY}_{keys.INDEX}'][[1,0]]
        setattr(heterodata[EDGE_KEY], keys.PERIODIC_INDEX, tensordict[f'{CENTER_KEY}_{keys.INDEX}'][2])
    # label
    for label in keys.LABELS.values():
        if label.key in tensordict:
            heterodata[label.parent][keys.REAL_PREFIX+label.key] = tensordict[label.key]
    return heterodata

def transform(
    npydict: Dict[str, np.ndarray],
    atom_cutoff: float,
    atom_sel: int,
    probe_cutoff: float,
    probe_sel: int,
    dtype: torch.dtype
):
    tensordict = npydict_to_tensordict(npydict, dtype)
    tensordict = construct_adjacency(tensordict, atom_cutoff, atom_sel, probe_cutoff, probe_sel)
    heterodata = tensordict_to_heterodata(tensordict)
    return heterodata
