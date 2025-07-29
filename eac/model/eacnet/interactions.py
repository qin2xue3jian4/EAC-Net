import torch
from typing import Dict, List

from torch_scatter import scatter
from e3nn import o3, nn
from e3nn.o3 import Irreps

from .base import MLPLayer, FullyConnected
from ...data import keys

MAX_ELEC_LAYER = 7
FULL_PATH = False

# function
def get_gate(
    node_feature_irreps_in: Irreps,
    edge_attr_irreps_in: Irreps,
    lmax: int,
    num_features: int,
    output_scalar: bool = False,
):
    p_acts = {
        1: torch.nn.functional.silu,
        -1: torch.tanh,
    }
    # Obtain the form of full multiplication based on the form of nodes and edges, and filter to obtain the final result form
    tp = o3.FullTensorProduct(
        node_feature_irreps_in,
        edge_attr_irreps_in
    )
    irreps_out_tmp = tp.irreps_out.simplify()
    irreps_scalars, act_scalars, irreps_gated = [], [], []
    physic_irreps = [irrep for __, irrep in edge_attr_irreps_in]
    for __, ir in irreps_out_tmp:
        l, p = ir.l, ir.p
        if ir not in physic_irreps:
            continue
        if l > lmax:
            continue
        if l == 0:
            if output_scalar and p == -1:
                continue
            elif output_scalar:
                irreps_scalars.append((num_features*10,(l,p)))
            else:
                irreps_scalars.append((num_features,(l,p)))
        elif not output_scalar:
            irreps_gated.append((num_features,(l,p)))
    irreps_scalars, irreps_gated = o3.Irreps(irreps_scalars), o3.Irreps(irreps_gated)
    # Input form excluding scalar part
    ir = o3.Irrep('0e') if o3.Irrep('0e') in [ir for __, ir in irreps_scalars] else o3.Irrep('0o')
    irreps_gates = o3.Irreps([(mul,ir) for mul, __ in irreps_gated])
    # activation function
    act_scalars = [ p_acts[ir.p] for _, ir in irreps_scalars ]
    act_gates = [ p_acts[ir.p] for _, ir in irreps_gates ] # modified
    # input: scalar(irreps_scalars+irreps_gates) + tensor(irreps_gated)
    # output: scalar(irreps_scalars2) + tensor(irreps_gated2)
    # irreps_scalars2 = act_scalars(irreps_scalars)
    # irreps_gated2 = act_gates(irreps_gates)*irreps_gated
    # length of irreps_gates and irreps_gated is the same.
    gate = nn.Gate(
        irreps_scalars = irreps_scalars,
        act_scalars=act_scalars,
        irreps_gates = irreps_gates,
        act_gates = act_gates,
        irreps_gated = irreps_gated,
    )
    irreps_in: o3.Irreps = gate.irreps_in
    irreps_out: o3.Irreps = gate.irreps_out
    return gate, irreps_in, irreps_out

def get_node_edge_tensorproduct(
    node_feature_irreps_in,
    edge_attr_irreps_in,
    gate_irreps_out,
):
    irreps_mid = []
    instructions = []
    physic_irreps = [irrep for __, irrep in edge_attr_irreps_in]
    for i, (mul1, ir_in) in enumerate(node_feature_irreps_in):
        for j, (mul2, ir_edge) in enumerate(edge_attr_irreps_in):
            for ir_out in ir_in * ir_edge:
                if not FULL_PATH and ir_out not in physic_irreps:
                    continue
                if ir_out in gate_irreps_out:
                    k = len(irreps_mid)
                    if FULL_PATH or ir_out.l == ir_edge.l or ir_out.l == ir_in.l:
                        irreps_mid.append((mul1, ir_out))
                        instructions.append((i, j, k, "uvu", True))
                    else:
                        irreps_mid.append((mul2, ir_out))
                        instructions.append((i, j, k, "uvv", True))
    irreps_mid = o3.Irreps(irreps_mid)
    irreps_mid, p, _ = irreps_mid.sort()

    # Permute the output indexes of the instructions to match the sorted irreps:
    instructions = [
        (i_in1, i_in2, p[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]
    node_tp_edge = o3.TensorProduct(
        node_feature_irreps_in, # node irreps
        edge_attr_irreps_in, # edge irreps
        irreps_mid, # result(edge) irreps to scatter
        instructions, # paths
        shared_weights=False,
        internal_weights=False,
    )
    irreps_out: o3.Irreps = node_tp_edge.irreps_out
    return node_tp_edge, irreps_out

# interaction layer
class BaseInteractionLayer(torch.nn.Module):
    def __init__(
        self,
        node_feature_irreps_in: Irreps,
        edge_attr_irreps_in: Irreps,
        weight_neuron: List[int],
        weight_active,
        feature_length=16,
        lmax=2,
        output_scalar=False,
        node_attr_length: int = None,
        feature_smooth: bool = False,
        resnet: bool = False,
        last_update: bool = True,
        edge_key: tuple = keys.ATOM_EDGE_KEY,
    ):
        super().__init__()
        # gate
        self.gate, self.gate_irreps_in, self.gate_irreps_out = get_gate(
            node_feature_irreps_in,
            edge_attr_irreps_in,
            lmax,
            feature_length,
            output_scalar,
        )
        # tensor product
        self.node_edge_tp, netp_irreps_out = get_node_edge_tensorproduct(
            node_feature_irreps_in,
            edge_attr_irreps_in,
            self.gate_irreps_out,
        )
        self.edge_neighbor_ratio = 20 ** 0.5
        # edge weight
        weight_neuron = weight_neuron + [self.node_edge_tp.weight_numel, ]
        self.edge_weight = FullyConnected(weight_neuron, weight_active)

        self.dst_feature_linear = MLPLayer(netp_irreps_out, self.gate_irreps_in, resnet=True)

        self.feature_smooth = feature_smooth
        self.resnet = resnet
        if resnet:
            self.node_update = o3.FullyConnectedTensorProduct(
                node_feature_irreps_in,
                f'{node_attr_length}x0e',
                self.gate_irreps_in,
            )
        
        self.last_update = last_update
        if last_update:
            self.last_feature_linear = MLPLayer(
                node_feature_irreps_in,
                node_feature_irreps_in,
                resnet=True
            )
        self.src, __, self.dst = self.edge = edge_key

class AtomInteractionLayer(BaseInteractionLayer):

    def forward(self, data: Dict):
        edge = data[self.edge]
        node_src, node_dst = edge[keys.INDEX]
        
        # source features
        src_node_features = data[self.src][keys.FEATURES]
        if self.last_update:
            src_node_features = self.last_feature_linear(src_node_features)
        src_to_edge_features = src_node_features[node_src]
        
        # source to edge
        edge_weight = self.edge_weight(edge[keys.BASIS_WEIGHT])
        edge_features = self.node_edge_tp(
            src_to_edge_features,
            edge[keys.ATTR],
            edge_weight
        )
        edge_features = edge_features.div(self.edge_neighbor_ratio)
        if self.feature_smooth:
            edge_features = edge_features * edge[keys.SMOOTH]
        
        # edge to destination
        dst_node_features = scatter(
            edge_features,
            node_dst,
            dim=0,
            dim_size=data[self.src][keys.POS].shape[0]
        )
        dst_node_features = self.dst_feature_linear(dst_node_features)
        
        # resnet
        if self.resnet:
            node_plus = self.node_update(data[self.dst][keys.FEATURES], data[self.dst][keys.ATTR])
            new_node_features = dst_node_features + node_plus
        
        # gate
        data[self.dst][keys.FEATURES] = self.gate(new_node_features)
        return data

class ProbeInteractionLayer(BaseInteractionLayer):
    
    def forward(self, data: Dict):
        edge = data[self.edge]
        
        # last edge features
        if keys.FEATURES in edge:
            last_grid_features = edge[keys.FEATURES]
        else:
            node_atom = edge[keys.INDEX][0]
            last_grid_features = data[keys.ATOM][keys.FEATURES][node_atom]
        
        if self.last_update:
            last_grid_features = self.last_feature_linear(last_grid_features)
        
        # edge tensor product
        edge_weight = self.edge_weight(edge[keys.BASIS_WEIGHT])
        grid_edge_features = self.node_edge_tp(
            last_grid_features,
            edge[keys.ATTR],
            edge_weight
        )
        grid_edge_features = grid_edge_features.div(self.edge_neighbor_ratio)
        if self.feature_smooth:
            grid_edge_features = grid_edge_features * edge[keys.SMOOTH]
        
        grid_edge_features = self.dst_feature_linear(grid_edge_features)
        
        # resnet
        if self.resnet:
            atom_index = edge[keys.INDEX][0]
            node_plus = self.node_update(last_grid_features, data[self.src][keys.ATTR][atom_index])
            grid_edge_features = grid_edge_features + node_plus
        
        # gate
        edge[keys.FEATURES] = self.gate(grid_edge_features)
        
        return data

class InteractionLayers(torch.nn.Sequential):
    def __init__(
        self,
        num_layers: int,
        lmax: int,
        edge_attr_irreps: Irreps,
        node_feature_irreps: Irreps,
        edge_feature_length: int,
        invariant_neurons: List[int],
        invariant_active: str,
        feature_length: int,
        ntype: str = 'atom',
        final_scalar = False,
        node_attr_length: int = None,
        feature_smooth: bool = False,
        resnet: bool = True,
        last_update: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.ntype = ntype
        invariant_neurons = [edge_feature_length,] + invariant_neurons
        
        layer_class = AtomInteractionLayer if ntype == 'atom' else ProbeInteractionLayer
        edge_key = keys.ATOM_EDGE_KEY if ntype == 'atom' else keys.PROBE_EDGE_KEY
        for inet in range(num_layers):
            output_scalar = final_scalar and inet == num_layers - 1
            single_layer = layer_class(
                node_feature_irreps_in=node_feature_irreps,
                edge_attr_irreps_in=edge_attr_irreps,
                weight_neuron=invariant_neurons,
                weight_active=invariant_active,
                feature_length=feature_length,
                lmax=lmax,
                output_scalar=output_scalar,
                node_attr_length=node_attr_length,
                feature_smooth=feature_smooth,
                resnet=resnet,
                last_update=last_update,
                edge_key=edge_key,
            )
            self.add_module(f'{ntype}_layer_{inet}', single_layer)
            node_feature_irreps = single_layer.gate.irreps_out
        self.node_irreps = node_feature_irreps
