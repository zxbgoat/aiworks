from torch import nn
from torch import Tensor
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer
from torch.utils.data import dataset


class LanguageModel(nn.Module):

    def __init__(self,
                 num_token:  int,
                 dim_model:  int,
                 num_head:   int,
                 dim_hidden: int,
                 num_layer:  int,
                 dropout:    float = 0.5):
        pass
