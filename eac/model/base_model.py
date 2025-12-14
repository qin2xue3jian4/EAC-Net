import re
import torch
from typing import Dict
from collections import OrderedDict
from ..utils.factory import BaseFactory

class ModelFactory(BaseFactory):
    _registry = {}

class BaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.infos: OrderedDict[str, float] = OrderedDict()

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        for key, value in self.infos.items():
            state_dict[f'infos.{key}'] = value
        return state_dict
    
    def safely_load_state_dict(self, state_dict: Dict, finetune: bool = False):
        msgs = []
        for name, params in self.named_parameters():
            if name in state_dict and params.shape == state_dict[name].shape:
                params.data.copy_(state_dict[name])
                if finetune and name.startswith('atom_env_model'):
                    params.requires_grad = False
            else:
                if name not in state_dict:
                    msgs.append(f'{name} not in state_dict')
                else:
                    msgs.append(f'{name} shape mismatch')
        for key, value in state_dict.items():
            if re.match(r'infos\..+', key):
                self.infos[key[6:]] = value
        return msgs
