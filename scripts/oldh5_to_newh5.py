import os
import glob
import tqdm
import h5py
import argparse
import numpy as np
from typing import List

from eac.data import keys

oldkey_to_newkey = {
    'chg': keys.CHARGE,
    'chgdiff': keys.CHARGE_DIFF,
    'atom_ids': keys.ATOM_TYPE,
    'coords': keys.ATOM_POS,
    'cells': keys.CELL,
    'energies': keys.ENERGY,
    'forces': keys.FORCE,
    'virials': keys.VIRIAL,
    'grids': keys.PROBE_POS,
}

def parse_argments():
    parser = argparse.ArgumentParser(description='Convert a directory to h5 files.')
    parser.add_argument('source_dir', type=str, help='Source directory.')
    parser.add_argument('target_dir', type=str, help='Target directory.')
    parser.add_argument('--depth', type=int, default=6, help='The depth of source data.')
    args = parser.parse_args()
    assert args.source_dir is not None
    assert args.target_dir is not None
    return args

def read_oldh5(old_h5_path: str):
    groups = {}
    with h5py.File(old_h5_path, 'r') as old_h5:
        for group_key, group in old_h5.items():
            nframe = group['cells'].shape[0]
            grid_shape = np.array(group['chg'].shape) if 'chg' in group else np.array([1, 10, 10, 10])
            new_group = {
                keys.PROBE_GRID_SHAPE: grid_shape,
                keys.FILE_SOURCE: old_h5_path,
            }
            for array_key, array_value in group.items():
                new_key = oldkey_to_newkey.get(array_key.replace('sample_', ''), array_key)
                if new_key in [keys.CHARGE, keys.CHARGE_DIFF]:
                    new_group[new_key] = array_value[:].reshape(nframe, -1)
                else:
                    new_group[new_key] = array_value[:]
            groups[group_key] = new_group
    return groups

def write_h5file(
    groups: List,
    h5_file: str
):
    with h5py.File(h5_file, 'w') as new_h5:
        for group_key, group in groups.items():
            new_h5.create_group(group_key)
            for item_key, item_value in group.items():
                if isinstance(item_value, np.ndarray):
                    new_h5[group_key].create_dataset(item_key, data=item_value)
                else:
                    new_h5[group_key].attrs[item_key] = item_value
    return None

def dir_to_files(source_dir: str, depth: int):
    if os.path.isfile(source_dir):
        if source_dir.endswith('.h5'):
            return [source_dir,]
        raise ValueError('Unsupported file type')
    files = []
    for depth in range(depth):
        path_pattern = os.path.join(source_dir, *(['*']*depth), '*.h5')
        for path in glob.glob(path_pattern):
            files.append(path)
    return files

def main():
    args = parse_argments()
    old_h5files = dir_to_files(args.source_dir, args.depth)
    target_dir = os.path.dirname(args.target_dir) if os.path.isfile(args.target_dir) else args.target_dir
    os.makedirs(target_dir, exist_ok=True)
    for ipath, old_h5file in enumerate(tqdm.tqdm(old_h5files)):
        groups = read_oldh5(old_h5file)
        base = os.path.basename(old_h5file)
        new_h5file = os.path.join(target_dir, base)
        write_h5file(groups, new_h5file)
    return None

if __name__ == '__main__':
    main()