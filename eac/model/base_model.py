import torch
from typing import Dict
from ..utils.factory import BaseFactory

class ModelFactory(BaseFactory):
    _registry = {}

class BaseModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def safely_load_state_dict(self, state_dict: Dict, finetune: bool = False):
        msgs = []
        for name, params in self.named_parameters():
            if name in state_dict and params.shape == state_dict[name].shape:
                params.data.copy_(state_dict[name])
                if finetune:
                    params.requires_grad = False
            else:
                if name not in state_dict:
                    msgs.append(f'{name} not in state_dict')
                else:
                    msgs.append(f'{name} shape mismatch')
        return msgs
    def forward(self, data: Dict, *args, **kwargs) -> Dict:
        pass