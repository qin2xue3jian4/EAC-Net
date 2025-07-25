import os
import tqdm
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.axes._axes import Axes

from ..data import keys
from ..losses.loss import MixedLoser
from .run import Controller, graph_to_labels

def save_3d_picture(
    points: np.ndarray,
    scalars: np.ndarray,
    picture_file: str,
):
    import plotly.graph_objects as go
    
    fig = go.Figure(data=go.Isosurface(
        x=points[:,0],
        y=points[:,1],
        z=points[:,2],
        value=scalars.flatten(),
        isomin=scalars.min(),
        isomax=scalars.max(),
        surface_count=30,
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorscale='Blackbody_r',
        colorbar=dict(
            title='density',
            thickness=20,
            len=0.75,
        ),
        opacity=0.8,
        lighting=dict(
            ambient=0.3,
            diffuse=0.8,
            specular=0.5,
            roughness=0.4
        ),
        lightposition=dict(x=100, y=100, z=100)
    ))

    # layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        margin=dict(t=30, l=0, b=0, r=0),
        width=900,
        height=700,
        title='Charge Density'
    )
    
    # save
    fig.write_html(f'{picture_file}.html', include_plotlyjs=True)
    return None

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
        # prepare
        ii, ll, pp = defaultdict(list), defaultdict(list), defaultdict(list)
        space_keys = [keys.CHARGE]
        if self.spin == 2:
            space_keys.append(keys.CHARGE_DIFF)
        last_frame_id, atom_representations = 'none', None
        ndata = len(self.loader)
        iframe = 0
        
        t1 = time.time()
        for idata, data in enumerate(tqdm.tqdm(self.loader, desc='Progress')):
            nprobe = data[keys.PROBE][keys.POS].shape[0]
            # atom representation
            if not self.shuffle:
                keep_same_frame = data[keys.FRAME_ID][0] == last_frame_id
                if not keep_same_frame and self.split and idata > 0:
                    self.show_result(t1, ii, ll, pp, iframe=iframe)
                    ii, ll, pp = defaultdict(list), defaultdict(list), defaultdict(list)
            
                last_frame_id = data[keys.FRAME_ID][0]
                if keep_same_frame and atom_representations is not None:
                    data[keys.ATOM]['features'] = atom_representations
                if not keep_same_frame:
                    iframe += 1
                    self._log(f'Testing frame {iframe}: {last_frame_id}')
            
            probe_empty = data[keys.PROBE][keys.POS].numel() == 0
            # empty probe
            if probe_empty:
                for key in space_keys:
                    preds[key] = torch.zeros((nprobe,), dtype=self.dtype, device=self.device)
            else:
                preds = self.model(data, out_type=self.out_type, return_atom_features=True)
                if not self.shuffle and not keep_same_frame:
                    atom_representations = preds[keys.ATOM_FEATURES]
            
            labels = graph_to_labels(data, preds.keys())
            for key in labels:
                ll[key].append(labels[key].detach())
                pp[key].append(preds[key].detach())
            ii[keys.PROBE].append(data[keys.PROBE][keys.POS].detach())
            
        self.show_result(t1, ii, ll, pp, ndata)
        
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
        # 三维图
        if keys.PROBE not in inputs or self.args.size is not None:
            return None
        try:
            grids = inputs[keys.PROBE].cpu().numpy()
            for value_type, value_dict in zip(['label', 'pred'], [labels, preds]):
                for key, value in value_dict.items():
                    array_value = value.cpu().numpy()
                    picture = os.path.join(self.output_dir, f'test_{value_type}_{key}_{iframe}')
                    save_3d_picture(grids, array_value, picture)
        except:
            self.logger.error('save 3d picture error')
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

