from turtle import forward
from typing import List, Tuple
from unittest import result
import torch
import torch.nn as nn
from torch import Tensor


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
        return out, hn

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
        e = torch.matmul(h_enc, torch.swapaxes(h_dec, 1, -1))
        a = torch.softmax(e, dim=1)
        c = torch.matmul(torch.swapaxes(a, 1, -1), h_enc)
        return c
