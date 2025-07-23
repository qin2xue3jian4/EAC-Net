import os
import time
import torch
import random
import functools
import numpy as np
from typing import List, Dict
from collections import (
    defaultdict,
    OrderedDict
)
from omegaconf import OmegaConf
import torch.distributed as dist

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
except ImportError:
    pass

from .run import Runner
from ..data import keys
from ..utils.envs import file_backups


flow_simply = {
    'train': 'trn',
    'valid': 'val',
    'test': 'tst',
}
key_simply = {
    keys.ENERGY: 'e',
    keys.FORCE: 'f',
    keys.VIRIAL: 'v',
    keys.CHARGE: 'c',
    keys.CHARGE_DIFF: 'd',
    keys.CHARGE_UP: 'u',
    keys.CHARGE_DOWN: 'd',
}

# move mean
def mean_step(data:np.array, margin=5):
    if isinstance(data, List):
        data = np.array(data)
    bsz = len(data)
    tmp = np.ones(margin, dtype=np.float64)
    long_data = np.concatenate((tmp*data[0], data, tmp*data[-1]))
    sum_data = np.zeros_like(data, dtype=np.float64)
    for istep in range(-margin, margin+1):
        sum_data += long_data[margin+istep:margin+istep+bsz]
    mean_data = sum_data / (2*margin+1)
    return mean_data

class Recorder(Runner):
    def __post_init__(self):
        super().__post_init__()
        self.flow_losses = defaultdict(list)
        self.times = {
            'run_start': None,
            'run_end': None,
            'epoch_start': None,
        }
        
        self.model_path = os.path.join(self.output_dir, 'models')
        self.latest = os.path.join(self.model_path, 'checkpoint.pt')
        os.makedirs(self.model_path, exist_ok=True)
        
        if self.args.plot:
            self.img_path = os.path.join(self.output_dir, 'imgs')
            os.makedirs(self.img_path, exist_ok=True)
    
    
    def create_lcurve_file(self, trainer):
        lcurve = os.path.join(self.output_dir, 'lcurve.out')
        if self.local_rank > 0:
            lcurve += f'.{self.local_rank}'
        if (old_lcurve := os.path.exists(lcurve)):
            with open(lcurve, 'r')as f:
                lines = f.readlines()
            file_backups(lcurve)
        self.lcurve_f = open(lcurve, 'w+', buffering=1)
        simplys = ['',] + [
            f'_{key_simply[loss_item]}'
            for loss_item in trainer.losser.KEYS
        ]
        first_line = '#       step'
        for ikey, key in enumerate(simplys):
            for flow in trainer.flows:
                title = f'rmse{key}_{flow_simply[flow]}'
                first_line += f'{title:>12s}'
            if ikey == 0:
                continue
            title = f'w{key}'
            first_line += f'{title:>12s}'
        first_line += f'{"lr":>12s}\n'
        second_line = '# If there is no available reference data, rmse_*_{val,trn} will print nan\n'
        self.lcurve_f.write(first_line)
        self.lcurve_f.write(second_line)
        if old_lcurve and lines[0] == first_line:
            for line in lines[2:]:
                this_step = int(line.split()[0])
                if this_step >= trainer.train_step:
                    break
                self.lcurve_f.write(line)
        return None
    
    def print_lcurve(self, trainer):
        if not hasattr(self, 'lcurve_f'):
            self.create_lcurve_file(trainer)
        lcurve_msg = f"{trainer.train_step:12d}"
        for flow in trainer.flows:
            lcurve_msg += f"{self.flow_losses[flow][-1]['loss']:12.3e}"
        ws = trainer.losser.ws
        for key, w in zip(trainer.losser.KEYS, ws):
            for flow in trainer.flows:
                value = self.flow_losses[flow][-1].get(key, 0.0)
                lcurve_msg += f'{value:12.3e}'
            lcurve_msg += f'{w:12.3e}'
        lcurve_msg += f'{trainer.lr:12.3e}\n'
        self.lcurve_f.write(lcurve_msg)
        self.lcurve_f.flush()
        return None
    
    def plot_loss(self, loss_keys: List[str]):
        
        # loss
        loss_keys = loss_keys + ['loss',]
        for key in loss_keys:
            fig, ax = plt.subplots()
            ax: Axes
            img_title = f'{key.title()} Loss vs Epoch'
            for flow_type, flow_losses in self.flow_losses.items():
                epoches = [l['epoch'] for l in flow_losses if key in l]
                if len(epoches) == 0:
                    continue
                item_losses = [l[key] for l in flow_losses if key in l]
                meaned_values = mean_step(item_losses, margin=self.cfg.record.margin)
                ax.plot(epoches, meaned_values, label=flow_type)
            if len(epoches) == 0:
                plt.close()
                continue
            ax.set_yscale(self.cfg.record.loss_yscale)
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax.tick_params(axis='x', which='both', top=True, bottom=True, labeltop=False, labelbottom=True)
            ax.tick_params(axis='y', which='both', right=True, left=True, labelright=False, labelleft=True)
            ax.legend()
            ax.set_title(img_title)
            fig.savefig(os.path.join(self.img_path, f'{key}-loss.png'))
            plt.close()
        
        # learning curve
        fig, ax = plt.subplots()
        ax: Axes
        img_title = 'Learning rate vs Epoch'
        epoches = [flow_loss['epoch'] for flow_loss in self.flow_losses['train']]
        lrs = [flow_loss['lr'] for flow_loss in self.flow_losses['train']]
        ax.plot(epoches, lrs, label='lr')
        ax.set_yscale('log')
        ax.set_xlabel('epoch')
        ax.set_ylabel('lr')
        ax.tick_params(axis='x', which='both', top=True, bottom=True, labeltop=False, labelbottom=True)
        ax.tick_params(axis='y', which='both', right=True, left=True, labelright=False, labelleft=True)
        ax.legend()
        ax.set_title(img_title)
        fig.savefig(os.path.join(self.img_path, 'lr.png'))
        plt.close()
        return
    
    def save_final(self, trainer):
        if hasattr(self, 'lcurve_f'):
            self.lcurve_f.close()
        final_file = os.path.join(self.model_path, f'{self.cfg.record.save_prefix}.{self.cfg.record.save_suffix}')
        model_state = {
            'model_state': trainer.module.state_dict(),
            'settings': OmegaConf.to_container(self.cfg, resolve=True),
        }
        torch.save(model_state, final_file)
        self._log(f'The model is save as {final_file}')
        return None
    
    def state_dict(self):
        state = OrderedDict()
        state['flow_losses'] = dict(self.flow_losses)
        npy_states = [
            npy_state.tolist() if isinstance(npy_state, np.ndarray) else npy_state
            for npy_state in np.random.get_state()
        ]
        state['random'] = {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
            "numpy": npy_states,
            "python": random.getstate(),
        }
        return state
    
    def load_state_dict(self, state_dict: OrderedDict):
        self.flow_losses = defaultdict(list, state_dict['flow_losses'])
        if 'random' in state_dict:
            random_state = state_dict['random']
            torch.set_rng_state(random_state["torch_cpu"])
            torch.cuda.set_rng_state_all(random_state["torch_cuda"])
            npy_states = tuple(
                np.array(npy_state, dtype=np.uint32) if isinstance(npy_state, torch.Tensor) else npy_state
                for npy_state in random_state["numpy"]
            )
            np.random.set_state(npy_states)
            random.setstate(random_state["python"])
        return None
    
    def save_checkpoint(self, epoch: int, trainer):

        loader_state_dict = {
            flow: loader.state_dict()
            for flow, loader in trainer.loaders.items()
        }
        
        model_state = {
            'settings': OmegaConf.to_container(self.cfg, resolve=True),
            'model_state': trainer.module.state_dict(),
            
            'optimizer': trainer.optimizer.state_dict(),
            'scheduler': trainer.scheduler.state_dict(),
            'losser': trainer.losser.state_dict(),
            'recorder': self.state_dict(),
            
            'start_epoch': epoch + 1,
            'train_step': trainer.train_step,
            
            'loaders': loader_state_dict,
        }
        
        filename = f'{self.cfg.record.save_prefix}-{epoch}.{self.cfg.record.save_suffix}'
        file = os.path.join(self.model_path, filename)
        torch.save(model_state, file)
        
        try:
            if os.path.exists(self.latest):
                os.remove(self.latest)
            os.link(file, self.latest)
        except:
            torch.save(model_state, self.latest)
            
        self._log(f'The checkpoint is saved as {file}.')
        
        return None
    
    def print_epoch_log(
        self,
        epoch: int,
        train_step: int,
        lr: float,
        loss_keys: List[str],
    ):
        used_time = time.time() - self.times['epoch_start']
        msg = f'Epoch {epoch} - step {train_step}: time: {used_time:.2f}s, lr={lr:.3e}, loss='
        
        # loss
        loss_str = [
            f'{flow_losses[-1]["loss"]:.3e}'
            for flow_losses in self.flow_losses.values()
        ]
        msg += '/'.join(loss_str)
        
        # item
        for key in loss_keys:
            if key not in self.flow_losses['train'][-1]:
                continue
            item_str = [
                f'{flow_losses[-1][key]:.3e}'
                for flow_losses in self.flow_losses.values()
            ]
            msg += f', {key}=' + '/'.join(item_str)
        
        msg += '.'
        self._log(msg)
        
        return
    
    def print_summary_log(self, epoch: int, trainer):
        self._log('# '*35)
        self._log(f'Epoch {epoch}/{trainer.nepoch}: lr = {trainer.lr:.3e}')
        
        # time
        time_used = time.time() - self.times['run_start']
        time_total = time_used / (epoch-trainer.start_epoch) * (trainer.nepoch-trainer.start_epoch)
        time_left = time_total - time_used
        if time_total > 3600 * 3:
            unit, unit_name = 3600, 'h'
        elif time_total > 3600:
            unit, unit_name = 60, 'min'
        else:
            unit, unit_name = 1, 's'
        time_used /= unit
        time_total /= unit
        time_left /= unit
        self._log(f'time(used/left/total): {time_used:.3f}/{time_left:.3f}/{time_total:.3f}{unit_name}')
        
        # flow loss
        for flow_type, flow_losses in self.flow_losses.items():
            if flow_losses[-1]['epoch'] != epoch:
                continue
            log_flow_losses = flow_losses[-self.cfg.record.log_freq:]
            
            # keys
            logs = set()
            for log_flow_loss in log_flow_losses:
                logs |= set(log_flow_loss.keys())
            logs = logs & set(trainer.losser.KEYS)
            logs = sorted(list(logs))
            
            # values
            total_loss = np.mean([l['loss'] for l in log_flow_losses])
            item_loss = [np.mean([l[key] for l in log_flow_losses if key in l]) for key in logs]
            item_str = '/'.join([f'{value:.3e}' for value in item_loss])
            
            flow_msg = f'{flow_type.title()}: total/{"/".join(logs)} loss - {total_loss:.3e}/' + item_str + '.'
            
            self._log(flow_msg)
        
        self._log('# '*35)
        return None
    
    
    def note_run(func):
        @functools.wraps(func)
        def wrapper(trainer, *args, **kwargs):
            
            self: Recorder = trainer.recorder
            self._log(f'Start to train.')
            
            flow_msg = ' + '.join([
                f'{flow}: {trainer.args.epoch_size or trainer.cfg.data[flow].epoch_size}'
                for flow in trainer.loaders
            ])
            # nepoch = self.cfg.nepoch
            # epoch_size = trainer.args.epoch_size or trainer.cfg.data['train'].epoch_size
            # frame_size = trainer.args.frame_size or trainer.cfg.data['train'].frame_size
            # size_msg = f'Nepoch {nepoch} x epoch size {epoch_size} x nframe {frame_size}'
            # if trainer.out_type != 'potential':
            #     probe_size = trainer.args.probe_size or trainer.cfg.data['train'].probe_size
            #     size_msg += f' x nprobe {probe_size}'
            self._log(f'Nepoch {self.cfg.nepoch} x epoch size ({flow_msg}).')
            
            if trainer.start_epoch > 1:
                self._log(f'Continuation mode, starting from epoch {trainer.start_epoch}')
            
            self.times['run_start'] = time.time()
            result = func(trainer, *args, **kwargs)
            self.times['run_end'] = time.time()
            
            if self.args.plot:
                self.plot_loss(trainer.losser.KEYS)
            
            self.save_final(trainer)
            for loader in trainer.loaders.values():
                loader.close()
            
            if dist.is_initialized():
                dist.destroy_process_group()
            
            used_time = self.times['run_end'] - self.times['run_start']
            self._log(f'Run end, used time: {used_time:.3f}s.')
            
            return result
        return wrapper

    def note_evaluate(func):
        def wrapper(trainer, epoch: int, flow_type: str, **kwargs):
            
            self: Recorder = trainer.recorder
            if flow_type == 'train':
                self.times['epoch_start'] = time.time()
            
            losses: Dict[str, List[float]] = func(trainer, epoch, flow_type, **kwargs)
            
            # record it
            flow_loss = {
                'epoch': epoch,
                'lr': trainer.lr,
                'step': trainer.train_step,
                **{
                    key: float(np.mean(value) if self.cfg.record.loss_func == 'mean' else value[-1])
                    for key, value in losses.items()
                }
            }
            self.flow_losses[flow_type].append(flow_loss)
            
            # show it
            if flow_type == trainer.flows[-1]:
                
                self.print_lcurve(trainer)
                self.print_epoch_log(epoch, trainer.train_step, trainer.lr, trainer.losser.KEYS)
                
                
                if epoch % self.cfg.record.log_freq == 0:
                    self.print_summary_log(epoch, trainer)
                
                if self.args.plot and epoch % self.cfg.record.plot_freq == 0:
                    self.plot_loss(trainer.losser.KEYS)
                
                if epoch % self.cfg.record.save_freq == 0:
                    self.save_checkpoint(epoch, trainer)
            return None
        return wrapper
    