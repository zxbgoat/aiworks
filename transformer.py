from typing import Any
from typing import Union
from typing import Optional
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import xavier_uniform_
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.container import ModuleList
from torch.nn.modules.normalization import LayerNorm


class Encoder(nn.Module):

    pass


class Transformer(nn.Module):

    def __init__(self,
                 dim_model: int = 512,
                 num_heads: int = 8,
                 num_encoders: int = 6,
                 num_decoders: int = 6,
                 dim_ffn: int = 2048,
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 custom_encoder: Optional[Any] = None,
                 custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = False,
                 norm_first: bool = False,
                 bias: bool = True,
                 device = None,
                 dtype = None
                ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                mem_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                mem_key_padding_mask: Optional[Tensor] = None,
                src_is_causal: Optional[bool] = None,
                tgt_is_causal: Optional[bool] = None,
                mem_is_causal: bool = False
                ) -> Tensor:
        pass
