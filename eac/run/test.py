import os
import torch
from typing import Dict
import matplotlib.pyplot as plt
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
                writer = Writer(self.args.save)
            
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
            
            # loss
            label = {key.replace('label_', ''): value for key, value in result.items() if key.startswith('label_')}
            preds = {key.replace('pred_', ''): value for key, value in result.items() if key.startswith('pred_')}
            loss, item_loss = self.losser(label, preds)
            loss_type, item_type = self.cfg.loss.probe_loss_func, self.cfg.loss.probe_item_func
            item_msg = ', '.join([f'{k}-{item_type}: {v:.3e}' for k,v in item_loss.items()])
            self._log(f'{frame_id}: loss-{loss_type}: {loss.item():.3e}, {item_msg}.')
            
            # plot
            if self.args.plot:
                self.plot(label, preds, frame_id)
            
            ngroup = self.loader.dataset.ngroups[ifile]
            if group_idx >= ngroup:
                base_filename = self.generate_filename(frame_id, istructure)
                file = os.path.join(self.output_dir, f'{base_filename}.{self.args.format}')
                writer.write_to_file(file)
                ifile += 1
                group_idx = 0
        return

    def plot(
        self,
        label: Dict[str, torch.Tensor],
        preds: Dict[str, torch.Tensor],
        frame_id: str,
    ):
        __, group_key, iframe = frame_id.split('|')
        # 对角线图
        for key, label_value in label.items():
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
            img = os.path.join(self.output_dir, f'test_{key}_{group_key}_{iframe}.png')
            fig.savefig(img)
            plt.close()
        return None


