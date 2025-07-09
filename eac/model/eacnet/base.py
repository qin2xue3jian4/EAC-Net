import torch
from e3nn import o3
from torch import (
    Tensor,
    Generator
)
from typing import (
    List,
    Union,
)
from e3nn.o3 import (
    Irreps,
    Instruction
)
from torch.nn import Module

def _no_grad_normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    generator: Generator = None,
):
    with torch.no_grad():
        return tensor.normal_(mean, std, generator=generator)

def is_scalar(irreps: Irreps) -> bool:
    return len(irreps) == 1 and irreps[0].ir == o3.Irrep('0e')

def get_active_fun(
    active: Union[str, Module, None] = None,
):
    if active is None or isinstance(active, Module) or active == '':
        active_fun = active
    elif active == 'leaky_relu':
        active_fun = torch.nn.LeakyReLU()
    elif active == 'relu':
        active_fun = torch.nn.ReLU()
    elif active == 'tanh':
        active_fun = torch.nn.Tanh()
    elif active == 'silu':
        active_fun = torch.nn.SiLU()
    else:
        active_fun = getattr(torch.nn, active)()
    return active_fun

class EquiModule(Module):
    
    irreps_in: Irreps
    irreps_out: Irreps
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    def _test_equivariance(self, ntest=20):
        origin_dtype = self.dtype
        self.to(torch.float64)
        inputs = torch.rand((ntest, self.irreps_in.dim), dtype=self.dtype, device=self.device)
        rotate = o3.rand_matrix()
        D_x = self.irreps_in.D_from_matrix(rotate).to(dtype=self.dtype, device=self.device)
        D_y = self.irreps_out.D_from_matrix(rotate).to(dtype=self.dtype, device=self.device)
        rotate_output = self.forward(torch.matmul(inputs, D_x))
        output_rotate = torch.matmul(self.forward(inputs), D_y)
        result = torch.allclose(rotate_output, output_rotate, 1E-5, 1E-5)
        self.to(origin_dtype)
        return result

    def _test_normal(self, ntest=200):
        inputs = torch.randn(ntest, self.irreps_in.dim, dtype=self.dtype, device=self.device)
        output = self.forward(inputs)
        mean = output.mean().item()
        std = output.std().item()
        return mean, std
    
    def _test(self):
        equivariance = self._test_equivariance()
        if not equivariance:
            print('Equivariance is wrong.')
        mean, std = self._test_normal()
        print(f'Module out mean is {mean:.3e}, std is {std:.3e}')
        return None

class EquiModules(EquiModule):
    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
    ):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.module_list = torch.nn.ModuleList()
    def append(self, module: EquiModule):
        self.module_list.append(module)
    def forward(self, inputs: Tensor) -> Tensor:
        for module in self.module_list:
            inputs = module(inputs)
        return inputs

class IdtEquiNet(EquiModule):
    def __init__(
        self,
        irreps_in: Irreps,
        idtmean: float = 0.1,
    ):
        super().__init__()
        self.irreps_in = self.irreps_out = irreps_in
        self.scalar = is_scalar(irreps_in)
        self.idt = torch.nn.Parameter(torch.empty(size=(irreps_in.num_irreps,)))
        _no_grad_normal_(self.idt, idtmean, 0.001)
    
    def forward(
        self,
        x: torch.Tensor,
    ):
        if self.scalar:
            return x * self.idt
        outputs = []
        idt_index, x_index = 0, 0
        for mul, ir in self.irreps_in:
            mul: int
            ir: o3.Irrep
            idt = torch.repeat_interleave(
                self.idt.narrow(-1, idt_index, mul),
                ir.dim,
                -1
            ) # (mul*dim,)
            output = x.narrow(-1, x_index, mul * ir.dim) * idt # (bsz, mul*dim)
            outputs.append(output)
            idt_index += mul
            x_index += mul * ir.dim
        outputs = torch.concatenate(outputs, dim=-1)
        return outputs
    
class ResNet(EquiModule):
    def __init__(
        self,
        modules: EquiModules,
        resnet: int = 1,
    ):
        super().__init__()
        self.resnet = resnet
        self.model = modules
        self.irreps_in = modules.irreps_in
        self.irreps_out = modules.irreps_out
        self.scalar = is_scalar(self.irreps_in) and is_scalar(self.irreps_out)
        
    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        if self.resnet == 1:
            res = x
        elif self.scalar:
            shape = (1,)*(x.dim()-1) + (self.resnet,)
            res = x.repeat(*shape)
        else:
            shape = (1,)*(x.dim()-1) + (self.resnet, 1)
            res = x.unsqueeze(-2).repeat(*shape).reshape(x.shape[:-1] + (-1,))
        y = self.model(x)
        y = y + res
        return y

def Linear(
    irreps_in: Irreps,
    irreps_out: Irreps,
    bias: bool = False,
    bias_std: float = 0.05,
    bias_mean: float = 0.0,
):
    def get_weight_std(in_features, out_features):
        base_std = torch.tensor(2.0/(in_features+out_features)).sqrt()
        return base_std# * modify_ratio
    
    scalar_linear = is_scalar(irreps_out) and is_scalar(irreps_in)
    if scalar_linear:
        linear = torch.nn.Linear(irreps_in[0].mul, irreps_out[0].mul, bias)
    else:
        linear = o3.Linear(irreps_in, irreps_out, biases=bias)

    if bias:
        _no_grad_normal_(
            linear.bias, mean = bias_mean, std = 1.0,
        )
        modify_ratio = (1-torch.tensor(bias_std).square()).sqrt()
    else:
        modify_ratio = 1.0

    if scalar_linear:
        in_features = linear.in_features
        out_features = linear.out_features
        weight_std = get_weight_std(in_features, out_features)
        _no_grad_normal_(
            linear.weight, std = weight_std,
        )
    else:
        index = 0
        for ins in linear.instructions:
            ins: Instruction
            if len(ins.path_shape) != 2:
                continue
            in_features, out_features = ins.path_shape
            num = in_features * out_features
            weight_std = get_weight_std(in_features, out_features)#shape_sum.sqrt() / ins.path_weight * base_ratio
            _no_grad_normal_(
                linear.weight[index:index+num], std = weight_std,
            )
            index += num
    return linear, scalar_linear

def MLPLayer(
    in_features: Union[int, Irreps],
    out_features: Union[int, Irreps],
    active: Union[None, Module, str] = None,
    bias: bool = True,
    idt: bool = False,
    resnet: bool = False,
    bias_std: float = 0.05,
    bias_mean: float = 0.0,
):
    # irreps
    irreps_in = Irreps(f'{in_features}x0e' if isinstance(in_features, int) else in_features)
    irreps_out = Irreps(f'{out_features}x0e' if isinstance(out_features, int) else out_features)
    irreps_in = irreps_in.simplify()
    irreps_out = irreps_out.simplify()
    modules = EquiModules(irreps_in, irreps_out)
    if len(irreps_in) == 1 and irreps_in[0].ir == o3.Irrep('0e'):
        in_features = irreps_in[0].mul
    if len(irreps_out) == 1 and irreps_in[0].ir == o3.Irrep('0e'):
        out_features = irreps_out[0].mul
    # linear
    linear, scalar_linear = Linear(irreps_in, irreps_out, bias, bias_std, bias_mean)
    modules.append(linear)
    
    # active
    if scalar_linear:
        active = get_active_fun(active)
        if active:
            modules.append(active)

    # idt
    if idt:
        if scalar_linear:
            if in_features == out_features:
                idtnet = IdtEquiNet(irreps_out)
                modules.append(idtnet)
        else:
            idtnet = IdtEquiNet(irreps_out)
            modules.append(idtnet)
    # resnet
    if resnet:
        resnet = 0
        if scalar_linear:
            if irreps_out[0].mul % irreps_in[0].mul == 0:
                resnet = irreps_out[0].mul // irreps_in[0].mul
        else:
            if len(irreps_in) == len(irreps_out):
                temp = 0
                for (ndim1, (l1, p1)), (ndim2, (l2, p2)) in zip(irreps_in, irreps_out):
                    if l1 == l2 and p1 == p2 and ndim2 % ndim1 == 0:
                        if temp == 0:
                            temp = ndim2 // ndim1
                        if ndim2 // ndim1 != temp:
                            break
                    else:
                        break
                else:
                    resnet = temp
        if resnet:
            modules = ResNet(modules, resnet)

    return modules

class FullyConnected(EquiModule):
    def __init__(
        self,
        neuron: List[Union[int, str, o3.Irreps]],
        active: Union[None, Module, str] = None,
        bias: Union[bool, List[bool]] = True,
        idt: Union[bool, List[bool]] = False,
        resnet: Union[bool, List[bool]] = False,
        final_active: Union[None, Module, str] = None,
        netname: str = 'FullyConnected',
        bias_std: float = 0.05,
        final_bias_mean: float = 0.0,
    ):
        super().__init__()
        assert len(neuron) > 1, 'The number of layers in a fully connected neural network should be greater than 1'
        self.mlplayers: List[EquiModule] = torch.nn.ModuleList()
        for inet, (h_in, h_out) in enumerate(zip(neuron, neuron[1:])):
            if inet != len(neuron) - 2:
                layer_active = active
            elif final_active is not None:
                layer_active = final_active
            else:
                layer_active = None
            biasi = bias if isinstance(bias, bool) else bias[inet]
            idti = idt if isinstance(idt, bool) else idt[inet]
            resneti = resnet if isinstance(resnet, bool) else resnet[inet]
            if inet == len(neuron) - 2:
                bias_mean = final_bias_mean
            else:
                bias_mean = 0.0
            self.mlplayers.append(
                MLPLayer(h_in, h_out, layer_active, biasi, idti, resneti, bias_std=bias_std, bias_mean=bias_mean)
            )
        self.irreps_in = self.mlplayers[0].irreps_in
        self.irreps_out = self.mlplayers[-1].irreps_out
        self.neuron = neuron
        self.active = active
        self.net_name = netname
        
    def __repr__(self):
        return f'{self.net_name}({self.neuron})'
    
    def forward(self, x) -> Tensor:
        for mlplayer in self.mlplayers:
            x = mlplayer(x)
        return x

class PolynomialSmooth(torch.nn.Module):

    ps: torch.Tensor
    es: torch.Tensor

    def __init__(
        self,
        p: int,
        rcut: float,
        epsilon: float = 1E-8,
    ):
        super().__init__()
        self.p = p
        self.rcut = rcut
        self.epsilon = epsilon
        ps = torch.tensor([p, p+1, p+2])
        es = torch.tensor([-(p+1)*(p+2)/2, p*(p+2), -p*(p+1)/2])
        self.register_buffer('ps', ps.unsqueeze(0))
        self.register_buffer('es', es.unsqueeze(0))
    
    def forward(
        self,
        lengths: torch.Tensor,
    ):
        safe_lengths = torch.clamp(lengths, min=self.epsilon, max=self.rcut)
        cutoff = 1 + (torch.pow(safe_lengths/self.rcut, self.ps)*self.es).sum(dim=-1, keepdim=True)
        return cutoff