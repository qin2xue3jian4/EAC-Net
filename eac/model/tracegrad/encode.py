import torch
from e3nn import o3
from typing import Dict, Tuple

from ...data import keys
from ...data.physics import (
    get_physics_encode,
    get_periodic_indexs
)
from .base import PolynomialSmooth

class AtomEncoding(torch.nn.Module):
    
    physics_encode: torch.Tensor
    
    def __init__(
        self,
        onehot: Dict,
        physics: Dict,
        feature_length: int,
    ):
        super().__init__()

        self.onehot_length = onehot.get('length', 0)
        self.onehot_opening = self.onehot_length > 0
        self.onehot_max_element = onehot.get('max_element', 110)
        if self.onehot_opening:
            self.onehot_net = torch.nn.Embedding(
                num_embeddings=self.onehot_max_element,
                embedding_dim=self.onehot_length,
            )
        
        physics_encode = get_physics_encode(physics)
        self.register_buffer('physics_encode', physics_encode)
        self.physics_length = physics_encode.shape[-1]

        self.attr_length = self.onehot_length + self.physics_length
        self.feature_length = feature_length
        self.linear = torch.nn.Linear(
            in_features=self.attr_length,
            out_features=feature_length,
            bias=True,
        )
        
    def forward(
        self,
        data: Dict,
        run_env: bool = True,
    ):
        atom_numbers = data[keys.ATOM]['type']
        
        if self.physics_length > 0:
            atom_attrs = physics_attrs = self.physics_encode[atom_numbers]
        
        if self.onehot_opening:
            atom_attrs = onehot = self.onehot_net(atom_numbers)
        
        if self.physics_length > 0 and self.onehot_opening:
            atom_attrs = torch.cat([physics_attrs, onehot], dim=-1)
        
        data[keys.ATOM][keys.ATTR] = atom_attrs
        
        if run_env:
            data[keys.ATOM][keys.FEATURES] = self.linear(atom_attrs)
        return

class BesselFunction(torch.nn.Module):
    def __init__(
        self,
        num_basis: int,
        rcut: float,
        epsilon: float,
    ):
        super().__init__()
        self.num_basis = num_basis
        self.rcut = rcut
        self.epsilon = epsilon
        kpir = torch.arange(1, num_basis+1).unsqueeze(0) * torch.pi / rcut
        self.register_buffer('kpir', kpir)
        
    def __call__(self, lengths: torch.Tensor):
        safe_lengths = torch.clamp(lengths, min=self.epsilon, max=self.rcut)
        bessel = torch.sin(safe_lengths*self.kpir) / (2*safe_lengths)
        return bessel

class GaussianFunction(torch.nn.Module):
    def __init__(
        self,
        num_basis: int,
        rcut: float,
        epsilon: float,
    ):
        super().__init__()
        self.num_basis = num_basis
        self.rcut = rcut
        self.epsilon = epsilon
        centers = torch.linspace(0, rcut, num_basis)
        self.register_buffer('centers', centers)
        self.centers = torch.nn.Parameter(torch.rand((num_basis,)))
        self.widths = torch.nn.Parameter(torch.rand((num_basis,)))
        
    def forward(
        self,
        lengths: torch.Tensor
    ):
        safe_lengths = torch.clamp(lengths, min=self.epsilon, max=self.rcut)
        features = torch.exp(-(self.centers - safe_lengths)**2 * self.widths)
        return features

class EdgeEncoding(torch.nn.Module):
    
    periodic_indexs: torch.Tensor
    
    def __init__(
        self,
        edge: Tuple,
        spherical: Dict,
        basis: Dict,
        smooth: Dict,
    ):
        super().__init__()
        self.src, __, self.dst = self.edge = edge

        self.lmax = spherical.get('lmax', 2)
        self.normalize = spherical.get('normalize', True)
        self.normalization = spherical.get('normalization', 'component')
        self.edge_sh_irreps = o3.Irreps.spherical_harmonics(lmax=self.lmax)
        self.sh = o3.SphericalHarmonics(
            self.edge_sh_irreps,
            normalize=self.normalize,
            normalization=self.normalization
        )

        basis_method = basis.pop('method', 'bessel')
        if basis_method == 'bessel':
            self.basis = BesselFunction(**basis)
        else:
            self.basis = GaussianFunction(**basis)
        
        self.smooth = PolynomialSmooth(**smooth)
        
        periodic_indexs = get_periodic_indexs()
        self.register_buffer('periodic_indexs', periodic_indexs, persistent=False)

    def forward(
        self,
        data: Dict[str, Dict[str, torch.Tensor]],
    ):
        edge = data[self.edge]
        node_src, node_dst = edge[keys.INDEX]
        
        # edge vector
        vectors = data[self.dst][keys.POS][node_dst] - data[self.src][keys.POS][node_src]
        if keys.PERIODIC_INDEX in edge:
            edge_periodic_index = self.periodic_indexs[edge[keys.PERIODIC_INDEX]]
            cells = data[keys.GLOBAL][keys.CELL]
            batch = data[self.src][keys.BATCH]
            src_cells = cells[batch[node_src]]
            edge_periodic_vectors = (src_cells*edge_periodic_index[:,:,None]).sum(dim=1)
            vectors -= edge_periodic_vectors
        if self.dst == 'atom':
            vectors = -vectors
        edge[keys.LENGTH] = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        
        # edge attr
        edge[keys.ATTR] = self.sh(vectors)
        
        # edge basis weight
        basis = self.basis(edge[keys.LENGTH])
        edge[keys.SMOOTH] = cutoff = self.smooth(edge[keys.LENGTH])
        edge[keys.BASIS_WEIGHT] = basis * cutoff
        
        return None