import math
from typing import List
from typing import Iterable
from timeit import default_timer as timer

from torchtext.datasets import multi30k
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


class Dataset:

    def __init__(self):
        self.src_lang = 'de'
        self.tgt_lang = 'en'
        self.src_tokens = get_tokenizer('spacy', language='de_core_news_sm')
        self.tgt_tokens = get_tokenizer('spacy', language='en_core_web_sm')


class PositEmbed(nn.Module):

    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000,
                ) -> None:
        super(PositEmbed, self).__init__()
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos_embed = torch.zeros(maxlen, emb_size)
        pos_embed[:, 0::2] = torch.sin(pos * den)
        pos_embed[:, 1::2] = torch.cos(pos * den)
        pos_embed = pos_embed.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embed', pos_embed)

    def forward(self, token_embed: Tensor) -> Tensor:
        return self.dropout(token_embed + self.pos_embed[:token_embed.size(0), :])


class TokenEmbed(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 embed_size: int
                ) -> None:
        super(TokenEmbed, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.emb_size = embed_size
    
    def forward(self, tokens: Tensor) -> Tensor:
        return self.embed(tokens.long()) * math.sqrt(self.emb_size)


class Translator(nn.Module):

    def __init__(self,
                 num_enc_layers: int,
                 num_dec_layers: int,
                 emb_size: int,
                 num_heads: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_ffn: int   = 512,
                 dropout: float = 0.1
                ) -> None:
        super(Translator, self).__init__()
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_embed = TokenEmbed(src_vocab_size, emb_size)
        self.tgt_embed = TokenEmbed(tgt_vocab_size, emb_size)
        self.pos_embed = PositEmbed(emb_size, dropout=dropout)
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=num_heads,
                                       num_encoder_layers=num_enc_layers,
                                       num_decoder_layers=num_dec_layers,
                                       dim_feedforward=dim_ffn,
                                       dropout=dropout)
    
    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_pad_mask: Tensor,
                tgt_pad_mask: Tensor,
                mem_key_pad_mask: Tensor
               ) -> None:
        pass

    def greedy_decode(src, src_mask, max_len, start_symbol):
        pass

    def translate(sentence: str):
        pass


class Trainer:

    def __init__(self,
                 model: nn.Module,
                 epochs: int,
                 batchsize: int):
        self.model     = model
        self.epochs    = epochs
        self.batchsize = batchsize
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __call__(self):
        for epoch in range(1, self.epochs+1):
            pass
    
    def collate_fn(self, batch):
        pass

    def train_epoch(self):
        self.model.train()
        losses = 0
        data_iter = Multi30k(split='train', language_pair=('de', 'en'))
        data_loader = DataLoader(data_iter, batch_size=self.batchsize, collate_fn=collate_fn)


if __name__ == '__main__':
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    BATCH_SIZE = 128
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
