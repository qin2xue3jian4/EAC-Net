import os
import tqdm
import h5py
import argparse
import numpy as np
from typing import Dict

from scripts.full_to_sample import convert_full_to_sample
from eac.data.read import file_paths_to_reader_modes, BaseReader, ReaderFactory, SpaceGroup

def parse_argments():
    parser = argparse.ArgumentParser(description='Convert a directory to h5 files.')
    parser.add_argument('source_dir', type=str, help='Source directory.')
    parser.add_argument('target_dir', type=str, help='Target directory.')
    parser.add_argument('--whole', action='store_true', help='Whether to store whole data in one file.')
    parser.add_argument('--wholefilename', type=str, default='whole', help='The whole filename.')
    parser.add_argument('--random', type=int, default=0, help='Number of random samples.')
    parser.add_argument('--grad', type=int, default=0, help='Number of grad samples.')
    parser.add_argument('--abs', type=int, default=0, help='Number of abs samples.')
    parser.add_argument('--sqrt', type=int, default=0, help='Number of sqrt samples.')
    parser.add_argument('--near', type=int, default=0, help='Number of near samples.')
    parser.add_argument('--depth', type=int, default=6, help='The depth of source data.')
    parser.add_argument('--random-max-cut', type=float, default=6.0, help='The maximum distance to cut.')
    parser.add_argument('--near-max-cut', type=float, default=1.2, help='The maximum distance to cut.')
    args = parser.parse_args()
    assert args.source_dir is not None
    assert args.target_dir is not None
    return args

def write_h5file(
    space_groups: Dict[str, SpaceGroup],
    h5_file: str
):
    with h5py.File(h5_file, 'w') as new_h5:
        for group_key, space_group in space_groups.items():
            new_h5.create_group(group_key)
            for item_key, item_value in space_group.group.items():
                if isinstance(item_value, np.ndarray):
                    new_h5[group_key].create_dataset(item_key, data=item_value)
                else:
                    new_h5[group_key].attrs[item_key] = item_value
            for attr_key, attr_value in space_group.extro_infos.items():
                new_h5[group_key].attrs[attr_key] = attr_value
    return None

def full_to_sample(
    reader: BaseReader,
    args: argparse.Namespace,
):
    random_nums = {
        key: getattr(args, key, 0)
        for key in ['random', 'grad', 'abs', 'sqrt', 'near']
    }

    sample = sum(random_nums.values()) > 0
    
    if sample:
        groups = {}
        for group_key, group in zip(reader.group_keys, reader.groups):
            for key in ['random', 'grad', 'abs', 'sqrt', 'near']:
                if random_nums[key] > 0:
                    groups[f'{group_key}_{key}'] = convert_full_to_sample(
                        group,
                        sample_num=random_nums[key],
                        sample_method=key,
                        random_max_cut=args.random_max_cut,
                        near_max_cut=args.near_max_cut,
                    )
    else:
        groups = {
            group_key: group
            for group_key, group in zip(reader.group_keys, reader.groups)
        }
    return groups

def main():
    args = parse_argments()
    target_dir = os.path.dirname(args.target_dir) if os.path.isfile(args.target_dir) else args.target_dir
    real_paths = file_paths_to_reader_modes([args.source_dir], search_depth=args.depth)
    
    os.makedirs(target_dir, exist_ok=True)
    whole_groups = {}
    for ipath, (real_path, reader_mode) in enumerate(
        tqdm.tqdm(real_paths.items())
    ):
        reader = ReaderFactory._registry[reader_mode](real_path)
        groups = full_to_sample(reader, args)
        if not args.whole:
            if os.path.isfile(real_path):
                base = '.'.join(os.path.basename(real_path).split('.')[:-1])
            else:
                base = os.path.basename(real_path)
            h5_file = os.path.join(target_dir, f'{base}.h5')
            write_h5file(groups, h5_file)
        else:
            for key, group in groups.items():
                whole_groups[f'{ipath}-{key}'] = group
    if args.whole:
        if os.path.isfile(args.target_dir):
            target_whole = args.target_dir
        else:
            target_whole = os.path.join(args.target_dir, args.wholefilename+'.h5')
        write_h5file(whole_groups, target_whole)
    return None

if __name__ == '__main__':
    main()