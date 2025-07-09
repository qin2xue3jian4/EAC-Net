import torch
from torch import Tensor
from typing import Dict, Tuple, List
from collections import defaultdict
from torch_geometric.data import HeteroData

from ..data import keys

def get_loss_func(loss_type: str):
    def mse_loss(diff: Tensor, label = None):
        return torch.square(diff).mean()
    def rmse_loss(diff: Tensor, label = None):
        return diff.square().mean().sqrt()
    def mae_loss(diff: Tensor, label = None):
        return diff.abs().mean()
    def ratio_loss(diff: Tensor, label: Tensor):
        return diff.abs().sum() / (label.abs().sum() + 1E-8)
    if loss_type == 'mse':
        return mse_loss
    elif loss_type == 'rmse':
        return rmse_loss
    elif loss_type in ['mae', 'l1', 'abs']:
        return mae_loss
    elif loss_type in ['ratio', 'rel']:
        return ratio_loss
    return rmse_loss

class MixedLoser(torch.nn.Module):

    KEYS: List[str]
    
    def __init__(
        self,
        cfg,
        device: torch.device,
        out_type: str,
    ):
        super().__init__()
        self.device = device
        self.spin_type = cfg.loss.spin_type
        self.space_negative_loss = cfg.loss.space_negative_loss
        
        self.KEYS = []
        origin_ws, end_ws = [], []
        if out_type in ['probe', 'mixed']:
            self.probe_loss_func = get_loss_func(cfg.loss.probe_loss_func)
            self.probe_item_func = get_loss_func(cfg.loss.probe_item_func)
            if cfg.model.probe.spin:
                if self.spin_type == 'origin':
                    self.KEYS.extend([keys.CHARGE, keys.CHARGE_DIFF])
                    origin_ws.extend([cfg.loss.charge_weight_start, cfg.loss.charge_diff_weight_start])
                    end_ws.extend([cfg.loss.charge_weight_end, cfg.loss.charge_diff_weight_end])
                else:
                    self.KEYS.extend([keys.CHARGE_UP, keys.CHARGE_DOWN])
                    origin_ws.extend([cfg.loss.charge_weight_start, cfg.loss.charge_weight_start])
                    end_ws.extend([cfg.loss.charge_weight_end, cfg.loss.charge_weight_end])
            else:
                self.KEYS.extend([keys.CHARGE])
                origin_ws.extend([cfg.loss.charge_weight_start,])
                end_ws.extend([cfg.loss.charge_weight_end,])
        
        if out_type in ['potential', 'mixed']:
            self.potential_loss_func = get_loss_func(cfg.loss.potential_loss_func)
            self.potential_item_func = get_loss_func(cfg.loss.potential_item_func)
            self.KEYS.append(keys.ENERGY)
            origin_ws.append(cfg.loss.energy_weight_start)
            end_ws.append(cfg.loss.energy_weight_end)
            if cfg.model.potential.grad:
                self.KEYS.extend([keys.FORCE, keys.VIRIAL])
                origin_ws.extend([cfg.loss.force_weight_start, cfg.loss.virial_weight_start])
                end_ws.extend([cfg.loss.force_weight_end, cfg.loss.virial_weight_end])
        
        self.method = cfg.loss.multy_weight_method
        if self.method is not None and self.method == 'auto':
            sigmas = (0.5 / torch.tensor(origin_ws)).sqrt()
            self.sigmas = torch.nn.Parameter(sigmas)
            self.sigmas.requires_grad = True
        else:
            self.lr = cfg.lr.start_lr if hasattr(cfg, 'lr') else 1.0
            self.origin_ws = torch.nn.Parameter(torch.tensor(origin_ws))
            self.origin_ws.requires_grad = False
            self.end_ws = torch.tensor(end_ws)
            self.alpha = (self.origin_ws-self.end_ws)/self.lr

    @property
    def ws(self):
        if self.method is not None and self.method == 'auto':
            return 0.5 / self.sigmas ** 2
        return self.end_ws + self.alpha * self.lr
    
    def forward(
        self,
        labels: Dict[str, torch.Tensor],
        preds: Dict[str, torch.Tensor],
        epoch: int=0,
        lr: float=1E-3
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        self.lr = lr
        loss_types = defaultdict()
        loss = torch.tensor(0.0, device=self.device)
        
        if keys.FORCE in self.KEYS:
            natom, ndim = labels[keys.FORCE].shape
        
        if keys.CHARGE_DIFF in preds and keys.CHARGE_DIFF in labels and (self.spin_type != 'origin' or self.space_negative_loss):
            for source in [preds, labels]:
                source[keys.CHARGE_UP] = (source[keys.CHARGE_DIFF] + source[keys.CHARGE])/2
                source[keys.CHARGE_DOWN] = (source[keys.CHARGE] - source[keys.CHARGE_DIFF])/2

        for ikey, (key, weight) in enumerate(zip(self.KEYS, self.ws)):
            if not (key in preds and key in labels):
                continue
            value_diff = preds[key].flatten() - labels[key].flatten()
            if key in [keys.ENERGY, keys.VIRIAL]:
                loss += self.potential_loss_func(value_diff) / natom * weight
                loss_types[key] = self.potential_item_func(value_diff, labels[key]).item() / natom
            elif key == keys.FORCE:
                loss += self.potential_loss_func(value_diff) * weight
                loss_types[key] = self.potential_item_func(value_diff).item()
            elif key in [keys.CHARGE, keys.CHARGE_DIFF, keys.CHARGE_UP, keys.CHARGE_DOWN]:
                loss += self.probe_loss_func(value_diff) * weight
                loss_types[key] = self.probe_item_func(value_diff, labels[key]).item()

        if keys.CHARGE in preds and keys.CHARGE in labels and self.space_negative_loss:
            if keys.CHARGE_DIFF in preds and keys.CHARGE_DIFF in labels:
                grid_scalars = [keys.GRID_SCALAR_UP, keys.GRID_SCALAR_DOWN]
            else:
                grid_scalars = [keys.CHARGE]
            for key in grid_scalars:
                negative_loss = torch.mean(torch.nn.functional.relu(-preds[key])) * 1.0
                loss += negative_loss

        if self.method is not None and self.method == 'auto':
            sigma_mul = torch.tensor(1.0, device=self.device)
            for ikey, key in enumerate(self.KEYS):
                if key in preds and key in labels:
                    sigma_mul *= self.sigmas[ikey]
            log_sigma = torch.log(sigma_mul)
            loss += log_sigma + (-log_sigma).item()
        
        return loss, loss_types
