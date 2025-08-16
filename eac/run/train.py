import os
import torch
import numpy as np
from typing import Dict
from collections import (
    defaultdict,
)
import torch.nn.utils as nn_utils

from ..tools import EMA, get_optimizer, Schedulers
from .record import Recorder
from ..data.load import LoaderWrapper
from ..losses.loss import MixedLoser
from .run import Controller, graph_to_labels

class Trainer(Controller):
    def __post_init__(self):
        super().__post_init__()
        
        self.get_loaders()
        
        self.recorder = Recorder(
            self.args,
            self.cfg
        )
        self.optimizer = get_optimizer(self.module, self.cfg)
        self.ema = EMA(self.module, self.cfg.run.ema_decay)
        self.scheduler = Schedulers(
            self.optimizer,
            self.cfg.run.schedulers
        )
        self.losser = MixedLoser(
            self.cfg,
            device=self.device,
            out_type=self.out_type
        )
        self.break_train = False
        self.load_state()
    
    def load_state(self):
        if self.method == 'checkpoint':
            for state_key, state_value in self.state_dict.items():
                try:
                    if state_key in ['optimizer', 'scheduler', 'losser', 'recorder', 'ema']:
                        getattr(self, state_key).load_state_dict(state_value)
                    elif state_key in ['start_epoch', 'train_step']:
                        setattr(self, state_key, state_value)
                    elif state_key == 'loaders':
                        for flow, loader in self.loaders.items():
                            assert flow in state_value
                            loader.load_state_dict(state_value[flow])
                except:
                    self._log(f'Failed to load state dict {state_key}.', 0, 'ERROR')
        else:
            if self.method == 'new' and not self.cfg.data.lazy_load:
                self.module.infos.update(self.loaders['train'].dataset.collect_infos())
            self.start_epoch = 1
            self.train_step = 0
        return
    
    def get_loaders(self):
        self._log('Loading datasets.')
        self.loaders: Dict[str, LoaderWrapper] = {}
        if self.cfg.data.exclude_keys is None:
            exclude_keys = None
        elif isinstance(self.cfg.data.exclude_keys, str):
            assert os.path.exists(self.cfg.data.exclude_keys)
            with open(self.cfg.data.exclude_keys) as f:
                exclude_keys = [line.strip() for line in f.readlines()]
        else:
            assert isinstance(self.cfg.data.exclude_keys, list)
            exclude_keys = self.cfg.data.exclude_keys
        for flow in ['train', 'valid', 'test']:
            if flow not in self.cfg.data or self.cfg.data[flow] is None:
                continue
            frame_size = self.args.frame_size or self.cfg.data[flow].frame_size
            probe_size = self.args.probe_size or self.cfg.data[flow].probe_size
            if self.args.num_workers is not None:
                num_workers = self.args.num_workers
            else:
                num_workers = self.cfg.data[flow].num_workers
            epoch_size = self.args.epoch_size or self.cfg.data[flow].epoch_size
            loader = self._load_paths_data(
                self.cfg.data[flow].paths,
                frame_size=frame_size,
                probe_size=probe_size,
                num_workers=num_workers,
                epoch_size=epoch_size,
                exclude_keys=exclude_keys,
            )
            nfile = len(loader.dataset.readers)
            ngroup = len(loader.dataset.groups)
            nbsz = loader.dataset.length
            batch_msg = f'{frame_size} frames'
            if self.out_type != 'potential':
                batch_msg += f' x {probe_size} probes'
            self._log(f'{flow} dataset loaded: {nfile} files, {ngroup} groups, {nbsz} batch x {batch_msg}.', 1)
            self.loaders[flow] = loader
        self.flows = list(self.loaders.keys())
        return
    
    @Recorder.note_run
    def run(self):
        self.nepoch = self.cfg.nepoch
        for epoch in range(self.start_epoch, self.nepoch+1):
            self.model.train()
            self.evaluate(epoch=epoch, flow_type='train')
            self.model.eval()
            self.ema.apply(self.module)
            self.evaluate(epoch=epoch, flow_type='valid')
            if 'test' in self.flows:
                self.evaluate(epoch=epoch, flow_type='test')
            self.ema.restore(self.module)
            if self.break_train:
                break
        return
    
    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']
    
    @Recorder.note_evaluate
    def evaluate(self, epoch: int, flow_type: str):
        
        losses = defaultdict(list)
        lr = self.lr
        need_grad = (
            flow_type == 'train' or 
            (hasattr(self.module, 'potential_pre_nets') and self.module.potential_pre_nets.grad)
        )

        with torch.set_grad_enabled(need_grad):
            for data in self.loaders[flow_type]:
                
                preds: Dict = self.model(data, out_type=self.args.out_type)
                labels = graph_to_labels(data, preds.keys())
                loss, item_loss = self.losser(labels, preds, self.train_step, lr)
                
                if flow_type == 'train':
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.cfg['grad_max_norm'] is not None:
                        nn_utils.clip_grad_norm_(self.model.parameters(), self.cfg['grad_max_norm'])
                    self.optimizer.step()
                    self.ema.update(self.module)
                    self.train_step += 1
                
                losses['loss'].append(loss.item())
                for key, value in item_loss.items():
                    losses[key].append(value)
        
        if flow_type == 'valid':
            epoch_mean_loss = np.mean(losses['loss'])
            next_ema_decay = 0.5 + 0.5 * np.exp(1-np.exp(min(epoch_mean_loss, 1.0)))
            self.ema.decay = next_ema_decay
            scheduler_change = self.scheduler.step(epoch, epoch_mean_loss)
            if scheduler_change:
                self._log(f'Scheduler changed to {self.scheduler.now_type}')
        
        return losses
