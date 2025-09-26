import torch
import numpy as np
from torch import Tensor
from torch_cluster import radius
from typing import Dict, List, Union
from torch_geometric.data import HeteroData

from .. import keys
from ..physics import get_periodic_indexs

DEFAULT_NODE_SEL = 40
periodic_indexs = get_periodic_indexs()

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
    atom_coord: Tensor,
    atom_numbers: Tensor,
    periodic_vectors: Tensor,
    rcut: float,
    sels: Union[int, List[int], None] = None,
    atom_types: Union[Tensor, List, None] = None,
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
def add_edge_index(
    heterodata: HeteroData,
    index_3d: torch.Tensor,
    edge_key: str,
):
    heterodata[edge_key][keys.INDEX] = index_3d[[1,0]]
    setattr(heterodata[edge_key], keys.PERIODIC_INDEX, index_3d[2])
    return

def construct_adjacency(
    heterodata: HeteroData,
    atom_cutoff: float,
    atom_sel: int,
    probe_cutoff: float,
    probe_sel: int,
):
    cell = heterodata[keys.GLOBAL][keys.CELL]
    periodic_vectors = torch.matmul(periodic_indexs.to(cell.dtype), cell)
    # atom
    atom_index_3d = atom_coord_to_index(
        heterodata[keys.ATOM][keys.POS],
        heterodata[keys.ATOM][keys.TYPE],
        periodic_vectors,
        atom_cutoff,
        atom_sel,
    )
    add_edge_index(heterodata, atom_index_3d, keys.ATOM_EDGE_KEY)
    # probe
    if keys.PROBE in heterodata.node_types:
        probe_index_3d = radius_3d(
            heterodata[keys.PROBE][keys.POS],
            heterodata[keys.ATOM][keys.POS],
            periodic_vectors,
            rcut=probe_cutoff,
            sel=probe_sel,
            skip_self_neighbor=False,
            strict_nearest=True,
        )
        add_edge_index(heterodata, probe_index_3d, keys.PROBE_EDGE_KEY)
    return

def npydict_to_heterodata(
    npydict: Dict[str, np.ndarray],
    dtype: torch.dtype,
):
    heterodata = HeteroData()
    heterodata[keys.ATOM][keys.POS] = torch.from_numpy(npydict[keys.ATOM_POS]).to(dtype)
    heterodata[keys.ATOM][keys.TYPE] = torch.from_numpy(npydict[keys.ATOM_TYPE]).to(torch.long)
    # probe
    if keys.PROBE_POS in npydict:
        heterodata[keys.PROBE][keys.POS] = torch.from_numpy(npydict[keys.PROBE_POS]).to(dtype)
        heterodata[keys.PROBE][keys.IDXS] = torch.from_numpy(npydict[f'{keys.PROBE}_{keys.IDXS}']).to(torch.long)
    # label
    for label in keys.LABELS.values():
        if label.key in npydict:
            heterodata[label.parent][keys.REAL_PREFIX+label.key] = torch.from_numpy(npydict[label.key]).to(dtype)
    # global
    for key, type_convert in keys.GLOBAL_KEYS.items():
        if key in npydict:
            if type_convert == 'long':
                heterodata[keys.GLOBAL][key] = torch.from_numpy(npydict[key]).to(torch.long)
            elif type_convert == 'dtype':
                heterodata[keys.GLOBAL][key] = torch.from_numpy(npydict[key]).to(dtype)
            else:
                heterodata[keys.GLOBAL][key] = npydict[key]
    return heterodata

def transform(
    npydict: Dict[str, np.ndarray],
    atom_cutoff: float,
    atom_sel: int,
    probe_cutoff: float,
    probe_sel: int,
    dtype: torch.dtype
):
    heterodata = npydict_to_heterodata(npydict, dtype)
    construct_adjacency(heterodata, atom_cutoff, atom_sel, probe_cutoff, probe_sel)
    return heterodata
