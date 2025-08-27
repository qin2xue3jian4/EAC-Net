import math
import torch
import numpy as np
from torch import Tensor
from torch.optim import Optimizer
from collections import OrderedDict
from omegaconf import DictConfig

from ..utils.factory import BaseFactory

class SchedulerFactory(BaseFactory):
    _registry = {}

class Scheduler:
    def __init__(
        self,
        optimizer: Optimizer,
        setting: DictConfig,
    ):
        self.optimizer = optimizer
        self.max_step = setting.nepoch
        self.last = self.start_step = 0

    def _muliple_rate(self, rate=1.0):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= rate
        return None
    
    def _get_lr(self, idx=0):
        return self.optimizer.param_groups[idx]['lr']
    
    def step(self, step, value):
        self._real_step(step, value)
        return step - self.start_step == self.max_step
    
    def _real_step(self, step, value):
        raise NotImplementedError("Subclasses must implement this method")
    
    def start(self, step):
        self.start_step = self.last = step
        return None

    def state_dict(self):
        state_dict = OrderedDict()
        state_dict['last'] = self.last # last update lr
        state_dict['start_step'] = self.start_step # start epoch
        if hasattr(self, 'rate'):
            state_dict['rate'] = self.rate
        if hasattr(self, 'values'):
            state_dict['values'] = torch.tensor(self.values)
        return state_dict

    def load_state_dict(self, state_dict: OrderedDict[str, Tensor]):
        self.last = state_dict.get('last', self.last)
        self.start_step = state_dict.get('start_step', self.start_step)
        if 'rate' in state_dict and hasattr(self, 'rate'):
            self.rate = state_dict['rate']
        if 'values' in state_dict and hasattr(self, 'values'):
            self.values = state_dict['values'].tolist()
        return None

@SchedulerFactory.register('stable')
class StableScheduler(Scheduler):
    def _real_step(self, step=None, value=None):
        pass

@SchedulerFactory.register('cosine')
class CosineScheduler(Scheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        setting: DictConfig,
    ):
        super().__init__(optimizer, setting)
        self.eta_min = setting.min_lr # minimum learning rate
        self.T_max = getattr(setting, 'T_max', self.max_step)  # length of the cycle
        self.warmup_steps = setting.warmup_steps
        self.cycle_end = self.T_max
        self.T_mult = setting.T_mult
        self.base_lr = None
        self.decay = setting.decay
        
    def _real_step(self, step, value=None):
        current_step = step - self.start_step
        if current_step < self.warmup_steps:
            rate = ((current_step / self.warmup_steps) if self.warmup_steps > 0 else 1.0) * self.base_lr / self._get_lr()
        else:
            end = current_step >= self.max_step
            eta_min = self.eta_min if self.eta_min is not None else self.base_lr / 100
            current_step -= self.warmup_steps
            current_T = cycle_end = self.T_max
            cycle_start = 0
            base_lr = self.base_lr
            while current_step >= cycle_end:
                cycle_start = cycle_end
                current_T = int(current_T * self.T_mult)  # update T
                cycle_end = cycle_start + current_T
                base_lr *= self.decay
                eta_min *= self.decay
            if not end:
                step_in_cycle = current_step - cycle_start
                cos_factor = (1 + math.cos(math.pi * step_in_cycle / current_T)) / 2
                rate = (eta_min + (base_lr - eta_min) * cos_factor) / self._get_lr()
            else:
                rate = base_lr / self._get_lr()
        self._muliple_rate(rate)
        self.last = step

    def start(self, step):
        super().start(step)
        self.base_lr = self._get_lr()
        return None
    
    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['base_lr'] = self.base_lr
        return state_dict

    def load_state_dict(self, state_dict: OrderedDict[str, torch.Tensor]):
        super().load_state_dict(state_dict)
        self.base_lr = state_dict['base_lr']

@SchedulerFactory.register('exp')
class ExpScheduler(Scheduler):
    """
    Multiply by a fixed number every certain number of steps until the specified learning rate or maximum number of steps is reached
    """
    def __init__(
        self,
        optimizer: Optimizer,
        setting: DictConfig,
    ):
        super().__init__(optimizer, setting)
        self.freq = setting.freq # update freq
        self.end_lr = setting.end_lr # end learning rate
        self.rate = getattr(setting, 'rate', None)

    def _real_step(self, step=None, value=None):
        if step - self.last >= self.freq:
            self._muliple_rate(self.rate)
            self.last = step
        return None

    def start(self, step):
        self.start_step = self.last = torch.tensor(step)
        if self.rate is None:
            self.rate = np.power(self.end_lr/self._get_lr(), self.freq/self.max_step).item()
        return None

@SchedulerFactory.register('plateau')
class PlateauScheduler(Scheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, setting):
        super().__init__(optimizer, setting)
        self.cooldown = setting.cooldown
        self.rate = getattr(setting, 'rate', None)
        self.values = [] # evaluate values
        self.length = setting.length
        self.best = np.inf

    def _real_step(self, step=None, value=None):
        self.values.append(float(value))
        # skip
        if (
            (len(self.values) < 2 * self.length) or # The runtime is less than the set starting length
            (step - self.last < self.cooldown) # Cooling in progress
        ):
            pass
        else:
            last_best = np.mean(self.values[-2*self.length:-self.length])
            now_value = np.mean(self.values[-self.length:])
            if now_value >= last_best:
                self._muliple_rate(self.rate)
            self.last = step
        return None
