import torch
from typing import Dict, List
from e3nn.o3 import Irreps

from .encode import (
    AtomEncoding,
    EdgeEncoding,
)
from .heads import (
    ChargeHead,
    PotentialFinalNet,
    ProbeEdgeWeightNet,
    ProbeEdgeFeatureSimple,
)
from ...data import keys
from .base import FullyConnected
from .interactions import InteractionLayers

class AtomEnvNet(torch.nn.Module):
    def __init__(
        self,
        atom_encoding: Dict,
        edge_encoding: Dict,
        message_passing: Dict,
    ):
        super().__init__()
        
        self.atom_encoding = AtomEncoding(**atom_encoding)
        
        self.edge_encoding = EdgeEncoding(
            **edge_encoding,
            edge=keys.ATOM_EDGE_KEY
        )
        
        self.message_passing = InteractionLayers(
            **message_passing,
            edge_attr_irreps=self.edge_encoding.edge_sh_irreps,
            node_feature_irreps=Irreps(f'{self.atom_encoding.feature_length}x0e'),
            edge_feature_length=self.edge_encoding.basis.num_basis,
            ntype='atom',
            final_scalar=False,
            node_attr_length=self.atom_encoding.attr_length,
        )
    
    def forward(
        self,
        data: Dict[str, Dict[str, torch.Tensor]],
        run_env: bool = True,
    ):
        self.atom_encoding(data, run_env)
        if run_env:
            self.edge_encoding(data)
            self.message_passing(data)
        return
    
class ProbeFitNet(torch.nn.Module):
    def __init__(
        self,
        edge_encoding: Dict,
        message_passing: Dict,
        feature_simple: Dict,
        probe_edge_weight: Dict,
        charge_head: Dict,
        atom_env_irreps: Irreps,
        spin: bool = False,
        node_attr_length: int = None,
    ):
        super().__init__()
        self.edge_encoding = EdgeEncoding(
            **edge_encoding,
            edge=keys.PROBE_EDGE_KEY
        )
        
        self.message_passing = InteractionLayers(
            **message_passing,
            edge_attr_irreps=self.edge_encoding.edge_sh_irreps,
            node_feature_irreps=atom_env_irreps,
            edge_feature_length=self.edge_encoding.basis.num_basis,
            ntype='probe',
            final_scalar=True,
            node_attr_length=node_attr_length,
        )
        
        self.feature_simple = ProbeEdgeFeatureSimple(
            **feature_simple,
            basis_weight_length=self.edge_encoding.basis.num_basis,
            origin_feature_length=self.message_passing.node_irreps.dim,
            node_attr_length=node_attr_length,
        )
        edge_feature_length = self.feature_simple.edge_feature_length
        
        self.edge_weight_net = ProbeEdgeWeightNet(
            **probe_edge_weight,
            edge_feature_length=edge_feature_length
        )
        
        self.charge_head = ChargeHead(
            **charge_head,
            edge_feature_length=edge_feature_length,
            spin=spin,
        )
    def forward(
        self,
        data: Dict,
    ):
        self.edge_encoding(data)
        self.message_passing(data)
        self.feature_simple(data)
        self.edge_weight_net(data)
        self.charge_head(data)
        return

class PotentialNet(torch.nn.Module):
    def __init__(
        self,
        grad: bool,
        point_neurons: List[int],
        point_active: str,
        atom_env_irreps: Irreps,
    ):
        super().__init__()
        neuron = [atom_env_irreps, ] + point_neurons + [1,]
        self.potential_net = FullyConnected(neuron, point_active)
        self.point_head = PotentialFinalNet(grad=grad)
        
    def load_infos(self, infos: Dict):
        for key, value in infos.items():
            params = getattr(self.point_head, key)
            params.data.copy_(value)
    
    def __repr__(self):
        neuron_msg = '-'.join([str(n) for n in self.neuron])
        return f'AtomicNet({neuron_msg})'

    def forward(self, data: Dict):
        node_features = data[keys.ATOM][keys.FEATURES]
        data['atomic_offset_energy'] = self.potential_net(node_features).squeeze()# * self.e_std
        output = self.point_head(data)
        return output

class PotentialPreNet(torch.nn.Module):
    def __init__(
        self,
        grad: bool = True,
    ):
        super().__init__()
        self.grad = grad
    def forward(
        self,
        data: Dict[str, Dict[str, torch.Tensor]],
    ):
        if self.grad:
            batch = data[keys.ATOM][keys.BATCH]
            pos = data[keys.ATOM][keys.POS]
            cell = data[keys.GLOBAL][keys.CELL]
            pos.requires_grad_(True)
            
            data[keys.DISPLACEMENT] = displacement = torch.zeros_like(cell)
            displacement.requires_grad_(True)
            
            symmetric_displacement = 0.5 * (displacement + displacement.transpose(-1, -2))
            data[keys.ATOM][keys.POS] = pos + torch.bmm(
                pos.unsqueeze(-2), torch.index_select(symmetric_displacement, 0, batch)
            ).squeeze(-2)
            data[keys.GLOBAL][keys.CELL] = cell + torch.bmm(
                cell, symmetric_displacement
            )
        
        return
