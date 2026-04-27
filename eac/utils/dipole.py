import numpy as np
from ..data.physics import ELEMENTS
from ..data.keys import ATOM_POS, ATOM_TYPE, CELL

e_to_debye = 4.80320425

def calculate_dipole_func(
    rs: np.ndarray, # (N, 3) or (N,)
    rho: np.ndarray, # (N,)
    # dv: float,
):
    if rs.ndim == 1:
        return np.sum(rs * rho)# * dv
    return np.sum(rs * rho[:,np.newaxis], axis=0)# * dv

def voxel_average(chg: np.ndarray) -> np.ndarray:
    return (
        chg
        + np.roll(chg, -1, axis=0)
        + np.roll(chg, -1, axis=1)
        + np.roll(chg, -1, axis=2)
        + np.roll(np.roll(chg, -1, axis=0), -1, axis=1)
        + np.roll(np.roll(chg, -1, axis=0), -1, axis=2)
        + np.roll(np.roll(chg, -1, axis=1), -1, axis=2)
        + np.roll(np.roll(np.roll(chg, -1, axis=0), -1, axis=1), -1, axis=2)
    ) * 0.125

def wrap_positions_to_nearest(positions, reference, cell):
    inv_cell = np.linalg.inv(cell)
    frac_pos = positions @ inv_cell
    frac_ref = reference @ inv_cell
    
    wrapped_frac = frac_pos.copy()
    for i in range(3):
        delta = frac_pos[:, i] - frac_ref[i]
        wrapped_frac[:, i] = frac_ref[i] + (delta - np.round(delta))
    wrapped_pos = wrapped_frac @ cell
    return wrapped_pos

# calculate dipole
def calculate_dipole_electron(
    cell: np.ndarray,
    chg: np.ndarray,
    center: np.ndarray,
    volume: float=None,
    degree: int=1,
) -> np.ndarray:
    nx, ny, nz = ngrid = chg.shape
    i_grid = np.arange(1,nx+1) / (nx+1)
    j_grid = np.arange(1,ny+1) / (ny+1)
    k_grid = np.arange(1,nz+1) / (nz+1)
    I, J, K = np.meshgrid(i_grid, j_grid, k_grid, indexing='ij')
    frac_coords = np.stack([I.ravel(), J.ravel(), K.ravel()], axis=1)
    grid_points = frac_coords @ cell
    wrapped_grid_points = wrap_positions_to_nearest(grid_points, center, cell)
    chg_flat = chg.flatten()
    if volume is None:
        volume = np.abs(np.linalg.det(cell))
    dV = volume / np.prod(ngrid)
    charges_at_grid = chg_flat * dV
    # calculation
    if degree == 1:
        dipole_electron = -calculate_dipole_func(
            rs = wrapped_grid_points - center,
            rho = charges_at_grid,
        )
    elif degree == 2:
        dipole_electron = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                rs = (wrapped_grid_points[:,i]-center[i]) * (wrapped_grid_points[:,j]-center[j])
                dipole_electron[i,j] = calculate_dipole_func(
                    rs = rs,
                    rho = charges_at_grid,
                )
    dipole_electron *= e_to_debye
    return dipole_electron
def calculate_atomic_dipole(chg: np.ndarray, cell: np.ndarray, ionic_position: np.ndarray):
    chg = voxel_average(chg)
    volume = np.abs(np.linalg.det(cell))
    # result
    charge = np.sum(chg) * volume / np.prod(chg.shape),
    dipole = calculate_dipole_electron(cell, chg, ionic_position, volume, degree=1),
    second = calculate_dipole_electron(cell, chg, ionic_position, volume, degree=2),
    return charge, dipole, second

def write_result_to_xyz(result: dict, group: dict, xyz_file: str, iframe: int=0):
    atom_pos  = group[ATOM_POS][iframe]   # (natom, 3)
    atom_ids = group[ATOM_TYPE][iframe]  # (natom,)
    atom_type = [ELEMENTS[i-1] for i in atom_ids]
    cell      = group[CELL][iframe]       # (3, 3)
    natom     = len(atom_type)

    prop_meta = {}   # name -> (ncol, dtype_char)
    for key, val_list in result.items():
        sample = np.atleast_1d(val_list[0])
        ncol   = sample.size
        dtype  = "I" if np.issubdtype(sample.dtype, np.integer) else "R"
        prop_meta[key] = (ncol, dtype)

    # Properties
    # Fixed column：id(I,1)  species(S,1)  pos(R,3)
    properties_parts = ["id:I:1", "species:S:1", "pos:R:3"]
    for key, (ncol, dtype) in prop_meta.items():
        properties_parts.append(f"{key}:{dtype}:{ncol}")
    properties_str = ":".join(properties_parts)

    lattice_str = " ".join(f"{v:.10g}" for v in cell.flatten())

    with open(xyz_file, "w") as f:
        f.write(f"{natom}\n")
        f.write(f'Lattice="{lattice_str}" Properties={properties_str}\n')
        for i in range(natom):
            parts = [str(i), str(atom_type[i])]
            # coords
            parts += [f"{atom_pos[i, j]:.10g}" for j in range(3)]
            # properties
            for key, val_list in result.items():
                arr = np.atleast_1d(val_list[i]).flatten()
                parts += [f"{v:.10g}" for v in arr]
            f.write(" ".join(parts) + "\n")
    return
