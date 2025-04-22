import math
import torch
import pywt
import random
from torch import Tensor
from torch.nn import init
from torch.nn import functional as F
from torch.nn.parameter import Parameter

# k = math.pi / 32 # 超越mlp
k = math.pi / 16 # 最好 超越mlp
position_encoding = [lambda x: torch.sin(k*x),
                     lambda x: torch.cos(k*x),
                     lambda x: torch.sin(2*k*x),
                     lambda x: torch.cos(2*k*x),
                     lambda x: torch.sin(3*k*x),
                     lambda x: torch.cos(3*k*x),
                     lambda x: torch.sin(4*k*x),
                     lambda x: torch.cos(4*k*x)]
                     
origin_norm = [lambda x: x,
               lambda x: x**2, 
               lambda x: torch.sqrt(x), 
               lambda x: x**3, 
               lambda x: x**(1/3), 
               lambda x: torch.log(x+1) / torch.log(torch.tensor(2.)),
               lambda x: torch.pow(2, x) - 1,
               lambda x: (torch.exp(x)-1) / (torch.exp(torch.tensor(1.))-1)]

class Nonlinear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    # 实验证明，random selection 是无效的
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, base_func=position_encoding, random_num = 8) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.base_func = base_func
        self.random_num = min(random_num, len(self.base_func))
        self.weight = Parameter(torch.empty((out_features, in_features * random_num), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    # def minMax_normalization(self, x):
    #     x_min = (x.min(dim=-1).values).min(dim=-1).values
    #     x_max = (x.max(dim=-1).values).max(dim=-1).values
    #     x = (x - x_min) / (x_max - x_min + 1e-8)
    #     return x

    def nonlinear_layer(self, x):
        #     funcs_index = random.sample(range(len(self.base_func)), self.random_num)
        #     funcs_index.sort()
        #     x_expand = torch.cat([self.base_func[index](x) for index in funcs_index], dim=-1)
        x_expand = torch.cat([self.base_func[index](x) for index in range(len(self.base_func))], dim=-1)
        assert not torch.isnan(x_expand).any(), "Expand function warning: Array contains NaN."
        assert not torch.isinf(x_expand).any(), "Expand function warning: Array contains Inf."
        return x_expand

    def forward(self, input: Tensor) -> Tensor: 
        x = F.linear(self.nonlinear_layer(input), self.weight, self.bias)
        return x
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, base_func={self.base_func}'
