import numpy as np
from scipy.ndimage import gaussian_gradient_magnitude, distance_transform_edt
from eac.data.read import SpaceGroup
from eac.data import keys

_INDEXS_3 = ((1,2), (2,0), (0,1))
def _get_height_dim(
    volume: float,
    cell: np.ndarray,
    idim: int,
):
    jdim, kdim = _INDEXS_3[idim]
    area = np.linalg.norm(np.cross(cell[jdim], cell[kdim]))
    return volume / area

def generate_distances(
    cell: np.ndarray,
    node_pos: np.ndarray,
    ngfs: np.ndarray,
    max_near: float,
):
    nx, ny, nz = ngfs
    # euclidean distance transform
    inv = np.linalg.inv(cell.T)
    pos_frac = np.dot(node_pos, inv) % 1.0
    idx = np.floor(pos_frac * ngfs).astype(int) % ngfs
    mask = np.zeros(ngfs, dtype=bool)
    mask[idx[:,0], idx[:,1], idx[:,2]] = True
    # periodic
    volume = np.abs(np.linalg.det(cell))
    rcut = min(max(3.0, np.power(volume/node_pos.shape[0], 1/3)), max_near)
    ns = [
        int((rcut / _get_height_dim(volume, cell, idim)) * ngfs[idim])
        for idim in range(3)
    ]
    pad = tuple((n,n) for n in ns)
    mask_wrap = np.pad(mask, pad, mode='wrap')
    lengths = np.linalg.norm(cell, axis=1)
    D_wrap = distance_transform_edt(
        (~mask_wrap).astype(float),
        sampling=[lengths[0]/nx, lengths[1]/ny, lengths[2]/nz]
    )
    dist = D_wrap[ns[0]:-ns[0], ns[1]:-ns[1], ns[2]:-ns[2]].flatten()
    return dist

def convert_full_to_sample(
    space_group: SpaceGroup,
    sample_num: int = 500,
    sample_method: str = 'random',
    random_max_cut: float = 6.0,
    near_max_cut: float = 1.2,
):
    group = space_group.group
    ngfs = group[keys.PROBE_GRID_SHAPE][1:]
    nframe = group[keys.PROBE_GRID_SHAPE][0]
    
    if sample_method == 'random':
        weights = np.zeros((nframe, np.prod(ngfs)))
        for ptr, (cell, node_pos) in enumerate(zip(group[keys.CELL], group[keys.ATOM_POS])):
            dist = generate_distances(cell, node_pos, ngfs, near_max_cut)
            weights[ptr, dist<=random_max_cut] = 1.0
    
    elif sample_method == 'grad':
        scalar_grad = gaussian_gradient_magnitude(group[keys.CHARGE].reshape(nframe, *ngfs), sigma=1)
        weights = np.abs(scalar_grad.reshape(nframe, -1))

    elif sample_method == 'abs':
        weights = np.abs(group[keys.CHARGE].reshape(nframe, -1))

    elif sample_method == 'sqrt':
        weights = np.sqrt(np.abs(group[keys.CHARGE].reshape(nframe, -1)))
    
    elif sample_method == 'near':
        weights = np.zeros((nframe, np.prod(ngfs)))
        for ptr, (cell, node_pos) in enumerate(zip(group[keys.CELL], group[keys.ATOM_POS])):
            dist = generate_distances(cell, node_pos, ngfs, near_max_cut)
            weight = 1 / np.power(dist + 0.01, 2)
            weight[dist>near_max_cut] = 0.0
            weights[ptr] = weight

    else:
        raise ValueError('Unknown sample method')
    
    weights = weights / weights.sum(axis=1)
    sample_indexs = np.stack(
        [
            np.random.choice(np.prod(ngfs), size=sample_num, p=weights[iframe],replace=False)
            for iframe in range(nframe)
        ],
        axis=0
    )
    
    sample_group = {
        key: value
        for key, value in group.items()
        if key not in [keys.CHARGE, keys.CHARGE_DIFF, keys.PROBE_POS]
    }
    
    probe_poses = []
    for iframe in range(nframe):
        probe_pos = space_group._get_probe_pos(iframe=iframe, igrids=sample_indexs[iframe])
        probe_poses.append(probe_pos)
    sample_group[keys.PROBE_POS] = np.stack(probe_poses, axis=0)
    
    sample_group[keys.CHARGE] = np.stack([
        group[keys.CHARGE][iframe, sample_indexs[iframe]]
        for iframe in range(nframe)
    ], axis=0)
    
    if keys.CHARGE_DIFF in group:
        sample_group[keys.CHARGE_DIFF] = np.stack([
            group[keys.CHARGE_DIFF][iframe, sample_indexs[iframe]]
            for iframe in range(nframe)
        ], axis=0)
    
    return SpaceGroup(sample_group, space_group.extro_infos)
