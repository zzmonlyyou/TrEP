from typing import Optional
from torch import nn, Tensor
import torch
import math
from torch.nn import Softmax, ReLU,TransformerDecoderLayer,TransformerDecoder,TransformerEncoderLayer,TransformerEncoder

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:

        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



class TransformerModel_no_softmax(nn.Module):

    def __init__(self,ip_dim:int, seq_len: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(ip_dim, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(int(d_model*seq_len), 2)

        self.readout = nn.Flatten()
        self.init_params()

    def init_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            else:
                for ll in layer.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()

    def forward(self, src: Tensor) -> Tensor:
        src = src.permute(1,0,2)
        src = self.encoder(src) #* math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.permute(1,0,2)
        output = self.readout(output)
        output = self.decoder(output)
        #output = self.op_act(output)
        return output


