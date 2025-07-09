
import logging
from torch import Tensor
from typing import List
from torch.optim import Optimizer
from collections import OrderedDict

from .base import SchedulerFactory, Scheduler

class Schedulers:
    def __init__(
        self,
        optimizer: Optimizer,
        settings: List,
    ):
        self.scheduler_types: List[str] = []
        self.scheduler_list: List[Scheduler] = []
        for i, setting in enumerate(settings):
            scheduler = SchedulerFactory.create(
                setting.type,
                optimizer=optimizer,
                setting=setting
            )
            self.scheduler_list.append(scheduler)
            self.scheduler_types.append(setting.type)
        self.scheduler_list[0].start(0)
        self.ptr = 0
    
    @property
    def now_type(self):
        return self.scheduler_types[self.ptr]
    
    @property
    def now_scheduler(self):
        return self.scheduler_list[self.ptr]
    
    def step(self, step: int, value: float):
        change = self.now_scheduler.step(step, value)
        change = change and self.ptr < len(self.scheduler_list) - 1
        if change:
            self.ptr += 1
            self.now_scheduler.start(step)
            logging.info(f'The scheduler rotates to the next category: {self.now_scheduler.__class__.__name__}')
        return change

    def state_dict(self):
        state_dict = OrderedDict()
        state_dict['ptr'] = self.ptr
        for i, scheduler in enumerate(self.scheduler_list):
            state_dict[f'scheduler_{i}'] = scheduler.state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict: OrderedDict[str, Tensor]):
        self.ptr = state_dict['ptr']
        for i, scheduler in enumerate(self.scheduler_list):
            this_state = state_dict[f'scheduler_{i}']
            scheduler.load_state_dict(this_state)
        return None
