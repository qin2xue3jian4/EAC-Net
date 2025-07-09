import os
import h5py
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def parse_argments():
    parser = argparse.ArgumentParser(description='Plot sample grid.')
    parser.add_argument('h5file', type=str, help='Source file path.')
    parser.add_argument('outdir', type=str, default='outputs', help='Output image path.')
    args = parser.parse_args()
    return args

def main():
    args = parse_argments()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    with h5py.File(args.h5file, 'r') as f:
        for i, (group_key, group) in enumerate(f.items()):
            grids = group['probe_pos'][:].reshape(-1, 3)
            ax.scatter(grids[:, 0], grids[:, 1], grids[:, 2], label=group_key)
    # 添加图例和标签
    ax.legend()
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')
    ax.set_title('多个3D点集的可视化')

    # 调整视角
    ax.view_init(elev=20, azim=45)  # 可以调整这些值来改变视角

    plt.tight_layout()
    img_file = os.path.join(args.outdir, '3d_points.png')
    fig.savefig(img_file)
    plt.close()
if __name__ == '__main__':
    main()