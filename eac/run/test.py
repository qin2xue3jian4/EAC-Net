import os
import tqdm
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.axes._axes import Axes

from ..data import keys
from ..tools import MixedLoser
from .run import Controller
from ..data.write import Writer

class Tester(Controller):
    def __post_init__(self):
        super().__post_init__()
        self.cfg.loss.space_spin_type = 'all'
        self.losser = MixedLoser(self.cfg, self.device, self.out_type)
        frame_size = 1 if self.out_type == 'probe' or self.args.frame_size is None else self.args.frame_size
        probe_size = self.args.probe_size or 50
        epoch_size = -1 if self.args.size == -1 else self.args.size // probe_size
        num_workers = self.args.num_workers or 0
        self.loader = self._load_paths_data(
            self.args.paths,
            frame_size=frame_size,
            probe_size=probe_size,
            num_workers=num_workers,
            epoch_size=epoch_size,
        )
        self.model.eval()
        self.shuffle = epoch_size != -1
        self.split = self.args.split and not self.shuffle
        
    def run_probe(self):
        ifile, group_idx = 0, 0
        for istructure, (frame_id, igroup, iframe, result) in enumerate(self.inference_probe_loader(
            self.loader,
            need_contribute=False,
            need_label=True
        )):
            base_filename = self.generate_filename(frame_id, istructure)
            if group_idx == 0:
                writer = Writer()
            if iframe == 0:
                group_idx += 1
                space_group = self.loader.dataset.groups[igroup]
                natom = space_group.group[keys.ATOM_POS].shape[1]
                ngfs = space_group.group[keys.PROBE_GRID_SHAPE][1:]
                nframe = self.loader.dataset.nframes[igroup]
                frame_results = []
            frame_results.append(result)
            if iframe == nframe - 1:
                writer.append_result(frame_id, space_group, frame_results, sample_probe=False, ngfs=ngfs)

            ngroup = self.loader.dataset.ngroups[ifile]
            if group_idx >= ngroup:
                base_filename = self.generate_filename(frame_id, istructure)
                file = os.path.join(self.output_dir, f'{base_filename}.{self.args.format}')
                writer.write_to_file(file)
                ifile += 1
                group_idx = 0
        return

    def show_result(self, t1, ii, ll, pp, iframe):
        t2 = time.time() - t1
        self._log(f'used time: {t2:.3f}s')
        for dict_value in [ii, ll, pp]:
            for key, value in dict_value.items():
                dict_value[key] = torch.cat(value, dim=0)
        loss, item_loss = self.losser(ll, pp)
        self._log(f'loss: {loss.item():.3e}')
        for key, value in item_loss.items():
            self._log(f'{key}: {value:.3e}')
        if self.args.save:
            self.save(ii, ll, pp, iframe)
        if self.args.plot:
            self.plot(ii, ll, pp, iframe)
    
    def plot(self, inputs, labels, preds, iframe):
        # 对角线图
        for key, label_value in labels.items():
            pred_value = preds[key]
            fig, ax = plt.subplots()
            ax: Axes
            ax.scatter(label_value.cpu().numpy(), pred_value.cpu().numpy(), s=10, label=key)
            xmin, xmax = label_value.min().item(), label_value.max().item()
            ymin, ymax = pred_value.min().item(), pred_value.max().item()
            xmin, xmax = min(xmin, ymin), max(xmax, ymax)
            ax.plot([xmin, xmax], [xmin, xmax], '--')
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((xmin, xmax))
            ax.set_xlabel('real')
            ax.set_ylabel('pred')
            ax.legend()
            ax.set_title(key)
            ax.tick_params(axis='x', which='major', top=True, bottom=True, labeltop=False, labelbottom=True)
            ax.tick_params(axis='y', which='major', right=True, left=True, labelright=False, labelleft=True)
            img = os.path.join(self.output_dir, f'test_{key}_{iframe}.png')
            fig.savefig(img)
            plt.close()
        return None
    
    def save(self, inputs, labels, preds, iframe):
        result = {
            **{
                key: value.detach().cpu()
                for key, value in inputs.items()
            },
            **{
                f'label_{key}': value.detach().cpu()
                for key, value in labels.items()
            },
            **{
                f'pred_{key}': value.detach().cpu()
                for key, value in preds.items()
            }
        }
        if self.args.format == 'pt':
            torch.save(result, os.path.join(self.output_dir, f'test_result_{iframe}.pt'))
        elif self.args.format == 'npy':
            for key, value in result.items():
                np.save(os.path.join(self.output_dir, f'test_result_{key}_{iframe}.npy'), value.numpy())
        return None

