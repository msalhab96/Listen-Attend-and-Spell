from typing import List, Tuple, Union
from numpy import size
import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass
import random

class Encoder(nn.Module):
    """Implements the listen part of the LAS model, where the input is 
    mel spectrogram and the output is the last hidden states

    Args:
        input_size (int): Number of mel filterbanks
        num_layers (int): Number of stacked BiLSTM layers
        hidden_size (int): The hidden state Dimension of each BiLSTM layer
        truncate (bool): whether to truncate the outputs or to pad by zeros 
        reduction_factor (int, optional) the Time space reduction factor.
        Defaults to 2.
    """
    def __init__(
            self, 
            input_size: int, 
            num_layers: int, 
            hidden_size: int,
            truncate: bool,
            reduction_factor=2
            ) -> None:
        super().__init__()
        assert reduction_factor > 0, 'reduction_factor should be > 0'
        self.truncate = truncate
        self.reduction_factor = reduction_factor
        self.layers = nn.ModuleList([
            nn.LSTM(
                input_size=hidden_size * 2 * reduction_factor if i != 0 else input_size,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=True
            )
            for i in range(num_layers)
        ])
    
    def forward(self, x: Tensor):
        out = x
        for i, layer in enumerate(self.layers, start=1):
            out, (hn, cn) = layer(out)
            if i != len(self.layers):
                out = self.change_dim(out)
        return out, hn, cn

    def is_valid_length(self, x: Tensor) -> Tuple[bool, int]:
        """Check if the given tensor is valid to be passed 
        to dimensionality reduction phase or not 

        Args:
            x (Tensor): The tensor to be validated of shape (B, T, H)

        Returns:
            Tuple[bool, int]: whether the length is valid or not and the mod
        """
        mod = x.shape[1] % self.reduction_factor
        return mod == 0, mod

    def change_dim(self, x: Tensor) -> Tensor:
        (b, t, h) = x.shape
        is_valid, mod = self.is_valid_length(x)
        if not is_valid:
            if self.truncate:
                n_truncates = t - self.reduction_factor * (
                    t // self.reduction_factor
                    )
                x = x[..., :-1 * n_truncates, :]
            else:
                zeros = torch.zeros(size=(b, self.reduction_factor - mod, h))
                x = torch.cat((x, zeros), dim=1)
                t += self.reduction_factor - mod
        return x.reshape(
            b, 
            t // self.reduction_factor, 
            h * self.reduction_factor
            )

class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, h_enc: Tensor, h_dec: Tensor) -> Tensor:
        e = torch.matmul(h_enc, h_dec.permute(1, 2, 0))
        a = torch.softmax(e, dim=1)
        c = torch.matmul(h_enc.permute(0, 2, 1), a)
        return c.permute(2, 0, 1)

class Decoder(nn.Module):
    def __init__(
            self, 
            vocab_size: int,
            embedding_dim: int, 
            hidden_size: int,
            ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        self.lstm = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                batch_first=True
            )
        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=vocab_size
            )
    def forward(
            self, 
            x: Tensor,
            last_h: Tensor,
            last_c: Tensor
            ):
        out = self.embedding(x)
        out, (h, c) = self.lstm(out, (last_h, last_c))
        out = self.fc(out)
        return out, h, c

class Model(nn.Module):
    def __init__(
            self, 
            enc_params: dict, 
            dec_params: dict
            ):
        super().__init__()
        self.encoder = Encoder(**enc_params)
        self.attention = Attention()
        self.decoder = Decoder(**dec_params)

    def forward(
            self, 
            x: Tensor, 
            sos_token_id: int,
            max_len: int,
            target: Tensor,
            teacher_forcing_prob: float
            ):
        h_enc, hn, cn = self.encoder(x)
        (n, b, h) = hn.shape 
        hn = hn.permute(1, 0, -1).reshape(1, -1, n * h)
        cn = cn.permute(1, 0, -1).reshape(1, -1, n * h)
        result = (torch.ones(size=(b, 1)) * sos_token_id).long()
        (out, h, c) = self.decoder(result, hn, cn)
        predictions = out
        result = torch.argmax(out, dim=-1)
        for t in range(max_len - 1):
            context = self.attention(h_enc, h)
            (out, h, cn) = self.decoder(result, context, cn)
            predictions = torch.cat((predictions, out), dim=1)
            if random.random() > teacher_forcing_prob:
                result = target[:, t:t+1] 
            else:
                result = torch.argmax(out, dim=-1)
        return predictions

if __name__ == '__main__':
    enc_params = {
            "input_size" : 40, 
            "num_layers" : 3, 
            "hidden_size" : 128,
            "truncate" : False,
            "reduction_factor" : 1
    }
    dec_params = {
            "vocab_size" : 100,
            "embedding_dim" : 512, 
            "hidden_size" : 256
    }
    model = Model(enc_params, dec_params)
    x = torch.randn(size=(3, 24, 40))
    y = torch.randint(0, 55, size=(3, 10))
    out = model(x, 5, 10, y, 0.5)
    print(out.shape)