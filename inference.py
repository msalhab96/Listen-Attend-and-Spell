from torch import Tensor
import torch
from data import AudioPipeline
from train import (
    get_tokenizer,
    load_model
)
from hprams import hprams

def get_top_candidates(x: Tensor, n: int):
    # x -> (B, 1, V)
    x = torch.squeeze(x, dim=1)
    indices = torch.topk(x, n, dim=-1, sorted=True).indices
    steps = torch.arange(0, x.shape[0] * x.shape[1], x.shape[1])
    indices_mask = steps + indices.T
    indices_mask = indices_mask.T.reshape(-1)
    indices_mask = torch.sort(indices_mask).values
    values = torch.index_select(x.view(-1), dim=0, index=indices_mask)
    return torch.sort(indices, dim=1).values.view(-1), values


def keep_top_seq(indices: Tensor, values: Tensor, n: int):
    # indices -> (B * beta * beta, v)
    # values -> (B * beta, 1)
    n_seq = n ** 2
    # batch_size = indices.shape[0] // n_seq
    batched_values = values.view(-1, n_seq)
    val, ind = torch.topk(batched_values, k=n, sorted=True)
    ind = torch.topk(batched_values, k=n, sorted=True).indices
    ind_mask = torch.arange(0, indices.shape[0], n_seq) + ind.T
    ind_mask = ind_mask.T.reshape(-1)
    res_ind = torch.sort(ind_mask).values
    val = torch.index_select(values, dim=0, index=res_ind)
    ind = torch.index_select(indices, dim=0, index=res_ind)
    return ind, val


def get_result(result: Tensor, prob: Tensor, n: int):
    batch_size = prob.shape[0] // n
    inc = torch.arange(0, prob.shape[0], n)
    result_indices = torch.argmax(prob.reshape(batch_size, -1), dim=1) + inc
    return torch.index_select(result, dim=0, index=result_indices)


class BeamSearch:
    def __init__(
            self, 
            beta: int, 
            model, 
            max_len: int, 
            eos_token_id: int,
            sos_token_id: int
            ):
        self.beta = beta
        self.max_len = max_len
        self.model = model
        self.eos_token = eos_token_id
        self.sos_token = sos_token_id

    def decode(self, x: Tensor):
        # TODO: Add length normalizer
        # TODO: use torch.log on the values
        h_enc, temp_result, hn, cn = self.model.init_pred(x, self.sos_token)
        # temp_result -> (B, 1, V)
        result, tot_prob = get_top_candidates(temp_result, self.beta)
        mask = torch.ones(result.shape[0] * self.beta, dtype=torch.bool)
        for i in range(self.max_len):
            last_pred = self.get_last_pred(result)
            temp_result, hn, cn = self.model.predict_next(h_enc, hn, cn, last_pred)
            ind, vals = get_top_candidates(temp_result, self.beta)
            result = result.view(-1, i+1).repeat((1, self.beta)).view(-1, i+1)
            result = torch.cat([result, torch.unsqueeze(ind, 1)], dim=1)
            mask = self.update_mask(mask, result)
            vals *= mask
            tot_prob = tot_prob.view(-1, 1).repeat((1, self.beta)).view(-1, 1)
            tot_prob = tot_prob + torch.unsqueeze(vals, 1)
            result, tot_prob = keep_top_seq(result, tot_prob, self.beta)
        if self.beta == 1:
            return result
        return get_result(result, tot_prob, self.beta)

    def update_mask(self, mask: Tensor, ind: Tensor):
        last_pred = self.get_last_pred(ind)
        return mask & (last_pred != self.eos_token)

    def get_last_pred(self, ind: Tensor):
        return ind[..., -1]


class Predictor(BeamSearch):
    def __init__(self) -> None:
        self.tokenizer = get_tokenizer()
        self.model = load_model()
        self.audio_pipeline = AudioPipeline()
        super().__init__(
            **hprams.beam_search,
            model=load_model(),
            sos_token_id=self.tokenizer.special_tokens.sos_token,
            eos_token_id=self.tokenizer.special_tokens.eos_token
        )

    def predict(self, file_path):
        x = self.audio_pipeline(file_path)
        return self.decode(x)
