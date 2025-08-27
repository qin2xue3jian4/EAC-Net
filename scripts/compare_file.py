import os
import h5py
import argparse
import numpy as np
from ase.calculators.vasp import VaspChargeDensity

def parse_argments():
    parser = argparse.ArgumentParser(description='Compare two files.')
    parser.add_argument('real_file', type=str, help='Label file.')
    parser.add_argument('pred_file', type=str, help='Pred file.')
    parser.add_argument('--plot', action='store_true', help='Whether plot the result.')
    parser.add_argument('--path', type=str, default='.', help='The path to store the image file.')
    
    args = parser.parse_args()
    return args

def load_data(file: str):
    data = {}
    if file.endswith('.h5'):
        chgs, chgdiffs = [], []
        with h5py.File(file, 'r')as f:
            group_keys = sorted(list(f.keys()))
            for group_key in group_keys:
                group = f[group_key]
                if 'chgdiff' in group:
                    chgdiffs.append(group['chgdiff'][:].flatten())
                chgs.append(group['chg'][:].flatten())
        if len(chgdiffs):
            data['chgdiff'] = np.concatenate(chgdiffs)
        data['chg'] = np.concatenate(chgs)
    else:
        vcd = VaspChargeDensity(file)
        data['chg'] = np.concatenate(vcd.chg).flatten()
        if len(vcd.chgdiff):
            data['chgdiff'] = np.concatenate(vcd.chgdiff).flatten()
    return data

def main():
    args = parse_argments()
    real_data = load_data(args.real_file)
    pred_data = load_data(args.pred_file)
    for key, pred_value in pred_data.items():
        if key not in real_data:
            continue
        real_value = real_data[key]
        assert real_value.shape == pred_value.shape, f'Shape of {key} not equal.'
        nmae = np.mean(np.abs(real_value-pred_value)) / np.mean(np.abs(real_value))
        if args.plot:
            import matplotlib.pyplot as plt
            plt.cla()
            plt.scatter(real_value, pred_value, label=key, color='r')
            xmin = min(real_value.min(), pred_value.min())
            xmax = max(real_value.max(), pred_value.max())
            plt.plot([xmin, xmax], [xmin, xmax], 'g--')
            plt.xlabel('real')
            plt.ylabel('pred')
            plt.xlim(xmin, xmax)
            plt.ylim(xmin, xmax)
            plt.title(f'{key} NMAE: {nmae:.3f}')
            plt.savefig(os.path.join(args.path, f'debug_{key}.png'))
            plt.close()
        print(f'{key} NMAE: {nmae:.3e}')
    return

if __name__ == '__main__':
    main()