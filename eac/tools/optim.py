import torch
from typing import Union, Dict
from omegaconf import DictConfig
import torch_optimizer as optim

def get_optimizer(
    module: torch.nn.Module,
    cfg: Union[DictConfig, Dict]
) -> torch.optim.Optimizer:
    trainable_params = [p for p in module.parameters() if p.requires_grad]
    optim_type = cfg.run.optim_type.title()
    optim_cls = getattr(torch.optim, optim_type) if hasattr(torch.optim, optim_type) else getattr(optim, optim_type)
    optimizer = optim_cls(
        trainable_params,
        lr=cfg.run.start_lr,
        weight_decay=cfg.run.weight_decay,
    )
    return optimizer