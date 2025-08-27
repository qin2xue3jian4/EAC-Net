import torch
from typing import Dict

from ...data import keys
from .components import (
    AtomEnvNet,
    ProbeFitNet,
    PotentialNet,
    PotentialPreNet
)
from ..base_model import (
    BaseModel,
    ModelFactory
)

@ModelFactory.register('eac')
class EACMixModel(BaseModel):
    def __init__(
        self,
        atom: Dict,
        probe: Dict = None,
        potential: Dict = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        # atom env
        self.atom_env_model = AtomEnvNet(**atom)
        self.atom_env_irreps = self.atom_env_model.message_passing.node_irreps
        
        # probe
        if probe is not None:
            self.probe_fit_model = ProbeFitNet(
                **probe,
                atom_env_irreps=self.atom_env_irreps,
                node_attr_length=self.atom_env_model.atom_encoding.attr_length,
            )
        
        if potential is not None:
            self.potential_pre_nets = PotentialPreNet(potential.get('grad', True))
            self.potential_nets = PotentialNet(
                **potential,
                atom_env_irreps=self.atom_env_irreps
            )
        
    def forward(
        self,
        data: Dict[str, Dict[str, torch.Tensor]],
        out_type: str = None,
        return_pred_dict: bool = True,
        return_atom_features: bool = False,
    ):
        run_probe = (out_type is None or out_type == 'probe') and hasattr(self, 'probe_fit_model')
        run_potential = (out_type is None or out_type == 'potential') and hasattr(self, 'potential_nets')
        
        output:  Dict[str, torch.Tensor] = {}
        
        if run_potential:
            self.potential_pre_nets(data)
        
        run_env = keys.FEATURES not in data[keys.ATOM] or run_potential
        self.atom_env_model(data, run_env)
        
        if run_probe:
            self.probe_fit_model(data)
        
        if run_potential:
            self.potential_nets(data)
        
        for pred_key, pred_label in keys.LABELS.items():
            if pred_label.key in data[pred_label.parent] and f'{pred_label.key}_std' in self.infos:
                data[pred_label.parent][pred_label.key] = data[pred_label.parent][pred_label.key] * self.infos[f'{pred_label.key}_std'] + self.infos[f'{pred_label.key}_mean']
        
        if return_pred_dict:
            for pred_key, pred_label in keys.LABELS.items():
                if pred_label.key in data[pred_label.parent]:
                    output[pred_key] = data[pred_label.parent][pred_label.key]
            if return_atom_features:
                output[keys.ATOM_FEATURES] = data[keys.ATOM][keys.FEATURES]

        return output