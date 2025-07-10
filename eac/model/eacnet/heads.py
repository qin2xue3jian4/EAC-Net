import torch
from torch import Tensor
from typing import List, Dict
from torch_scatter import scatter

from ...data import keys, physics
from .base import FullyConnected
from .base import PolynomialSmooth

ELEMENTS_LENGTH = len(physics.ELEMENTS)

class ProbeEdgeWeightNet(torch.nn.Module):
    def __init__(
        self,
        method: str,
        edge_feature_length: int,
        active: torch.nn.Module = torch.nn.SiLU(),
        probe_cut: float = 6.0,
        neurons: List[int] = [240, 240],
        threshold: float = 1E-8
    ):
        super().__init__()
        self.method = method
        self.probe_cut = probe_cut
        self.cat_length = edge_feature_length
        self.threshold = threshold
        if self.method in ['exp+smooth', 'exp*smooth', 'softmax-smooth']:
            self.weight_neurons = [self.cat_length*2,] + neurons + [1,]
            self.edge_weight_net = FullyConnected(
                self.weight_neurons,
                active,
                bias=True
            )
        
    def forward(self, data: Dict[str, Tensor]):
        edge = data[keys.PROBE_EDGE_KEY]
        
        if self.method == 'one':
            edge[keys.WEIGHT] = 1.0
            return data
        elif self.method == 'smooth':
            edge[keys.WEIGHT] = edge[keys.SMOOTH]
            return data
        
        nprobe = data[keys.PROBE][keys.NUM_NODES]
        node_probe = edge[keys.INDEX][1]
        
        simpled_edge_features = edge[keys.FEATURES]
        probe_features = scatter(
            simpled_edge_features*edge[keys.SMOOTH],
            node_probe,
            0,
            dim_size=nprobe
        )[node_probe]
        cat_edge_features = torch.concatenate(
            [probe_features, simpled_edge_features],
            dim=-1
        )
        # feature for final weight
        edge_weight = self.edge_weight_net(cat_edge_features)
        if self.method == 'exp+smooth':
            edge_weight = edge_weight + edge[keys.SMOOTH]
        edge_weight = edge_weight - edge_weight.max()
        # # softmax
        if self.method == 'exp*smooth':
            exp_edge_weight = torch.exp(edge_weight) * edge[keys.SMOOTH] + self.threshold
        else:
            exp_edge_weight = torch.exp(edge_weight) + self.threshold
        sum_exp_edge_weight = scatter(exp_edge_weight , node_probe, 0, dim_size=nprobe)
        if self.method == 'softmax-smooth':
            edge[keys.WEIGHT] = exp_edge_weight / sum_exp_edge_weight[node_probe] * edge[keys.SMOOTH]
        else:
            edge[keys.WEIGHT] = exp_edge_weight / sum_exp_edge_weight[node_probe]
        return data

class Scalar2Charge(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, scalar: Tensor):
        return scalar

class ChargeHead(torch.nn.Module):
    def __init__(
        self,
        edge_feature_length: int,
        neuron: List[int],
        active: torch.nn.Module = torch.nn.SiLU(),
        spin: bool = False,
        bias: bool = True,
        idt: bool = True,
        resnet: bool = True,
    ):
        super().__init__()
        self.neuron = [edge_feature_length,] + neuron + [(2 if spin else 1),]
        self.spin = spin
        self.charge_net = FullyConnected(
            self.neuron,
            active,
            bias=bias,
            idt=idt,
            resnet=resnet,
            netname='charge_net',
        )
        self.scalar2charge = Scalar2Charge()
        
    def forward(
        self,
        data: Dict[str, Tensor],
    ):
        nprobe = data[keys.PROBE][keys.NUM_NODES]
        edge = data[keys.PROBE_EDGE_KEY]
        node_probe = edge[keys.INDEX][1]
        
        edge_scalars = self.charge_net(edge[keys.FEATURES])
        edge_charges = self.scalar2charge(edge_scalars)
        edge[keys.CHARGE] = edge_weighted_scalars = edge_charges * edge[keys.WEIGHT]
        probe_charges = scatter(
            edge_weighted_scalars,
            node_probe,
            dim=0,
            dim_size=nprobe
        )
        if self.spin:
            data[keys.PROBE][keys.CHARGE] = torch.sum(probe_charges, dim=-1)
            data[keys.PROBE][keys.CHARGE_DIFF] = probe_charges[:,0] - probe_charges[:,1]
        else:
            data[keys.PROBE][keys.CHARGE] = probe_charges[:,0]
        return

class ProbeEdgeFeatureSimple(torch.nn.Module):
    def __init__(
        self,
        origin_feature_length: int,
        drop_probability: float = 0.05,
        basis_weight_length: int = 8,
        neuron: List[int] = [1000, 240],
        cat_keys: List[str] = [],
        node_attr_length: int = None,
    ):
        super().__init__()
        
        self.dropout = torch.nn.Dropout(drop_probability)
        
        self.edge_feature_simple_net = FullyConnected(
            [origin_feature_length,] + neuron,
            torch.nn.SiLU(),
            bias=True
        )
        
        self.cat_keys = []
        self.edge_feature_length = neuron[-1]
        for cat_key in cat_keys:
            match cat_key:
                case 'length':
                    self.cat_keys.append([keys.PROBE_EDGE_KEY, keys.LENGTH])
                    self.edge_feature_length += 1
                case 'node_attrs':
                    self.cat_keys.append([keys.ATOM, keys.ATTR])
                    self.edge_feature_length += node_attr_length
                case 'basis':
                    self.cat_keys.append([keys.PROBE_EDGE_KEY, keys.BASIS_WEIGHT])
                    self.edge_feature_length += basis_weight_length
                case _:
                    raise ValueError(f'cat_key {cat_key} is not supported.')

    def forward(self, data: Dict[str, Tensor]):
        
        edge = data[keys.PROBE_EDGE_KEY]
        
        edge_features = self.dropout(edge[keys.FEATURES])
        
        edge_features = self.edge_feature_simple_net(edge_features)
        
        tensors_to_cat = [edge_features,]
        for cat_parent, cat_key in self.cat_keys:
            if cat_parent == keys.ATOM:
                atom_index, __ = edge[keys.INDEX]
                value = data[cat_parent][cat_key][atom_index]
            else:
                value = data[cat_parent][cat_key]
            tensors_to_cat.append(value)
        edge[keys.FEATURES] = torch.concatenate(tensors_to_cat, dim=-1)
        
        return data

class PotentialFinalNet(torch.nn.Module):
    def __init__(
        self,
        grad: bool = True
    ):
        super().__init__()
        bias_e = torch.zeros((ELEMENTS_LENGTH,))
        self.register_buffer('potential_bias', bias_e)
        e_std = torch.ones((1,))
        self.register_buffer('potential_std', e_std)
        total_e = torch.zeros((1,))
        self.register_buffer('potential_total', total_e)
        self.grad = grad
        
    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ):
        nbatch = data[keys.GLOBAL][keys.CELL].shape[0]
        atomic_offset_energy = data['atomic_offset_energy'] * getattr(self, 'potential_std')
        atomic_bias_energy = getattr(self, 'potential_bias')[data[keys.ATOM][keys.TYPE]]
        atomic_energy = atomic_bias_energy + atomic_offset_energy
        data[keys.GLOBAL][keys.ENERGY] = total_energy = scatter(
            atomic_energy,
            data[keys.ATOM][keys.BATCH],
            dim=0,
            dim_size=nbatch,
            reduce='sum',
        ) + getattr(self, f'potential_total')
        
        # data[keys.GLOBAL][keys.ENERGY] = total_energy.sum()
        
        if self.grad:
            grads = torch.autograd.grad(
                outputs=[total_energy.sum()],
                inputs=[data[keys.ATOM][keys.POS], data[keys.DISPLACEMENT]],
                create_graph=True,
                allow_unused=True
            )
            data[keys.ATOM][keys.FORCE] = -grads[0]
        
            volume = torch.linalg.det(data[keys.GLOBAL][keys.CELL]).abs().view(-1, 1, 1)
            data[keys.GLOBAL][keys.VIRIAL] = grads[1] / volume
        
        return
