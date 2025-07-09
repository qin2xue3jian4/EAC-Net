import h5py
import glob
import numpy as np

def convert_old_to_new_h5(old_h5_path: str, new_h5_path: str):
    groups = {}
    with h5py.File(old_h5_path, 'r') as old_h5:
        for group_key, group in old_h5.items():
            nframe = group['chg'].shape[0]
            new_group = {'full_shape': np.array(group['chg'].shape)}
            for array_key, array_value in group.items():
                if array_key in ['chg', 'chgdiff']:
                    new_group[array_key] = array_value[:].reshape(nframe, -1)
                else:
                    new_group[array_key] = array_value[:]
            groups[group_key] = new_group
    with h5py.File(new_h5_path, 'w') as new_h5:
        for group_key, group in groups.items():
            new_h5.create_group(group_key)
            for array_key, array_value in group.items():
                new_h5[group_key].create_dataset(array_key, data=array_value)
    print(f"Converted {old_h5_path} to {new_h5_path}")
    for group_key, group in groups.items():
        for array_key, array_value in group.items():
            print(f"{group_key}/{array_key}: {array_value.shape}")
    return None

if __name__ == "__main__":
    old_h5_path = '/bohr/c-si-90points-yhxo/v2/point/0_00.h5'
    new_h5_path = '/home/qinxuejian/projects/eac-net/datas/csi/0_00_new.h5'
    convert_old_to_new_h5(old_h5_path, new_h5_path)