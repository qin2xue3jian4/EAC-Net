import torch
from typing import Dict
from ..utils.factory import BaseFactory

class ModelFactory(BaseFactory):
    _registry = {}

class BaseModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def solid_atom_envnet(self, solid=True):
        pass
    def forward(self, data: Dict, *args, **kwargs) -> Dict:
        pass