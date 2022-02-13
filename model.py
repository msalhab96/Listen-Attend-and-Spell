from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor, device
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
        self.hidden_size = hidden_size
        self.truncate = truncate
        self.device = device
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
                zeros = zeros.to(self.device)
                x = torch.cat((x, zeros), dim=1)
                t += self.reduction_factor - mod
        return x.reshape(
            b,
            t // self.reduction_factor,
            h * self.reduction_factor
            )



class Attention(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            attention_size: int
            ):
        super().__init__()
        self.psi_fc = nn.Linear(hidden_size, attention_size)
        self.phi_fc = nn.Linear(hidden_size, attention_size)

    def forward(self, h_enc: Tensor, h_dec: Tensor) -> Tensor:
        psi_out = self.psi_fc(h_enc)
        phi_out = self.phi_fc(h_dec)
        e = torch.matmul(psi_out, phi_out.permute(1, 2, 0))
        a = torch.softmax(e, dim=1)
        c = torch.matmul(h_enc.permute(0, 2, 1), a)
        return c.permute(0, 2, 1)



class Decoder(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            enc_hidden_size: int,
            hidden_size: int,
            n_layers: int,
            device='cuda'
            ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        ).to(device)
        self.layers = nn.ModuleList([
            nn.LSTM(
                input_size=embedding_dim + enc_hidden_size \
                     if i == 0 else hidden_size,
                hidden_size=hidden_size,
                batch_first=True
            ).to(device)
            for i in range(n_layers)
        ])

        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=vocab_size
            ).to(device)

    def forward(
            self,
            x: Tensor,
            context: Tensor,
            last_h: Tensor,
            last_c: Tensor
            ):
        # x -> (b, 1)
        # context -> (b, 1, enc_h)
        # last_h -> (1, b, dec_h)
        # last_c -> (1, b, dec_h)
        out = self.embedding(x)  # (b, 1, emb)
        out = torch.cat([context, out], dim=-1)  # (b, 1, emb + enc_h)
        for i, layer in enumerate(self.layers):
            if i == 0:
                out, (h, c) = layer(out, (last_h, last_c))
            else:
                out, (h, c) = layer(out)
        out = self.fc(out)
        return out, h, c


class Model(nn.Module):
    def __init__(
            self,
            enc_params: dict,
            dec_params: dict,
            att_params: dict,
            device='cuda'
            ):
        super().__init__()
        self.device = device
        self.encoder = Encoder(**enc_params).to(device)
        self.attention = Attention(**att_params).to(device)
        self.decoder = Decoder(
            **dec_params,
            enc_hidden_size=enc_params['hidden_size'] * 2
            )

    def forward(
            self,
            x: Tensor,
            sos_token_id: int,
            max_len: int,
            target: Tensor,
            teacher_forcing_prob: float
            ):
        h_enc, out, hn, cn = self.init_pred(x, sos_token_id)
        predictions = out
        result = torch.argmax(out, dim=-1)
        for t in range(max_len - 1):
            out, hn, cn = self.predict_next(h_enc, hn, cn, result)
            predictions = torch.cat((predictions, out), dim=1)
            if random.random() > teacher_forcing_prob:
                result = target[:, t:t+1]
            else:
                result = torch.argmax(out, dim=-1)
        return torch.softmax(predictions, dim=2)

    def init_pred(self, x: Tensor, sos_token_id: int):
        h_enc, hn, cn = self.encoder(x)
        b = hn.shape[1]
        hn, cn = self.get_zeros_states(b)
        context = self.get_zeros_context(b)
        x0 = self.get_seed_tensor(sos_token_id, b)
        (out, h, c) = self.decoder(x0, context, hn, cn)
        return h_enc, out, h, c

    def predict_next(self, h_enc: Tensor, hn: Tensor, cn: Tensor, xt: Tensor):
        context = self.attention(h_enc, hn)
        (out, hn, cn) = self.decoder(xt, context, hn, cn)
        return out, hn, cn

    def get_zeros_states(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        return (
            torch.zeros(1, batch_size, self.decoder.hidden_size).to(self.device),
            torch.zeros(1, batch_size, self.decoder.hidden_size).to(self.device)
        )

    def get_zeros_context(self, batch_size: int) -> Tensor:
        return torch.zeros(batch_size, 1, self.encoder.hidden_size * 2).to(self.device)

    def get_seed_tensor(self, sos_token_id: int, batch_size: int):
        result = (torch.ones(size=(batch_size, 1)) * sos_token_id).long()
        return result.to(self.device)

    def enc_predict(self, x: Tensor):
        return self.encoder(x)
