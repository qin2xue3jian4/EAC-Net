import torch
import numpy as np
from typing import Dict
from collections import (
    defaultdict,
)
import torch.nn.utils as nn_utils

from .record import Recorder
from ..data.load import LoaderWrapper
from ..losses.loss import MixedLoser
from .run import Controller, graph_to_labels
from ..schedulers.scheduler import Schedulers

class Trainer(Controller):
    def __post_init__(self):
        super().__post_init__()
        
        self.get_loaders()
        
        self.recorder = Recorder(
            self.args,
            self.cfg
        )
        self.optimizer = torch.optim.Adam(
            self.module.parameters(),
            lr=self.cfg.lr.start_lr,
        )
        self.scheduler = Schedulers(
            self.optimizer,
            self.cfg.lr.schedulers
        )
        self.losser = MixedLoser(
            self.cfg,
            device=self.device,
            out_type=self.out_type
        )
        
        self.load_state()
    
    def load_state(self):
        if self.method == 'checkpoint':
            for state_key, state_value in self.state_dict.items():
                if state_key in ['optimizer', 'scheduler', 'losser', 'recorder']:
                    getattr(self, state_key).load_state_dict(state_value)
                elif state_key in ['start_epoch', 'train_step']:
                    setattr(self, state_key, state_value)
                elif state_key == 'loaders':
                    for flow, loader in self.loaders.items():
                        assert flow in state_value
                        loader.load_state_dict(state_value[flow])
        else:
            self.start_epoch = 1
            self.train_step = 0
        return
    
    def get_loaders(self):
        self._log('Loading datasets.')
        self.loaders: Dict[str, LoaderWrapper] = {}
        for flow in ['train', 'valid', 'test']:
            if flow not in self.cfg.data:
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
            self.evaluate(epoch=epoch, flow_type='valid')
            if 'test' in self.flows:
                self.evaluate(epoch=epoch, flow_type='test')
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
                    self.train_step += 1
                
                losses['loss'].append(loss.item())
                for key, value in item_loss.items():
                    losses[key].append(value)
        
        if flow_type == 'valid':
            epoch_mean_loss = np.mean(losses['loss'])
            scheduler_change = self.scheduler.step(epoch, epoch_mean_loss)
            if scheduler_change:
                self._log(f'Scheduler changed to {self.scheduler.now_type}')
        
        return losses