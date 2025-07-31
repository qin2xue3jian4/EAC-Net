import torch
from collections import OrderedDict

class EMA:
    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.999
    ):
        self.decay = decay
        self.shadow = OrderedDict()
        self.original = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_avg = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                self.shadow[name] = new_avg.clone()

    def apply(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.original[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.original[name])
    
    def load_state_dict(self, shadow: dict):
        for key, value in shadow.items():
            self.shadow[key].copy_(value)