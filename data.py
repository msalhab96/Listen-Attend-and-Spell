import torch
from torch import Tensor
from pathlib import Path
from typing import List, Tuple, Union
from tokenizer import ITokenizer
from torch.utils.data import Dataset
from data_loaders import IDataLoader
from hprams import hprams
from torchaudio.transforms import (
    Resample,
    MelSpectrogram
)
from utils import (
    IPipeline,
    load_audio
    )


class AudioPipeline(IPipeline):
    """Loads the audio and pass it through different transformation layers
    """
    def __init__(self) -> None:
        super().__init__()

    def run(
            self,
            audio_path: Union[str, Path],
            *args,
            **kwargs
            ) -> Tensor:
        x, sr = load_audio(audio_path)
        x = self._get_resampler(sr)(x)
        x = self._get_mel_spec_transformer()(x)
        x = x.permute(0, 2, 1)
        return x

    def _get_resampler(self, sr: int):
        return Resample(sr, hprams.data.sampling_rate)

    def _get_mel_spec_transformer(self):
        return MelSpectrogram(
            hprams.data.sampling_rate,
            n_mels=hprams.data.n_mel_channels
            )


class TextPipeline(IPipeline):
    """pass the text through different transformation layers
    """
    def __init__(self) -> None:
        super().__init__()

    def run(
            self,
            text: str
            ) -> str:
        text = text.lower()
        text = text.strip()
        return text


class Data(Dataset):
    def __init__(
            self,
            text_loader: IDataLoader,
            text_pipeline: IPipeline,
            audio_pipeline: IPipeline,
            tokenizer: ITokenizer,
            batch_size: int,
            max_len: int,
            descending_order=False
            ) -> None:
        super().__init__()
        self.data = self.prepocess_lines(text_loader.load().split('\n'))
        self.text_pipeline = text_pipeline
        self.audio_pipeline = audio_pipeline
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.max_size = []
        self.lengths = [
            self.__get_aud(aud_path).shape[1]
            for (aud_path, *_) in self.data
        ]
        self.data = sorted(
            zip(self.data, self.lengths),
            key=lambda x: x[-1],
            reverse=descending_order
            )
        self.lengths = sorted(self.lengths, reverse=descending_order)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        (aud_path, text), *_ = self.data[index]
        return (
            self.__get_padded_aud(aud_path, index),
            self.__get_padded_tokens(text)
            )

    def __get_aud(self, aud_path: Union[str, Path]) -> Tensor:
        aud = self.audio_pipeline.run(aud_path)
        assert aud.shape[0] == 1, f'expected audio of 1 channels got \
            {aud_path} with {aud.shape[0]} channels'
        return aud

    def __get_padded_aud(
            self,
            aud_path: Union[str, Path],
            aud_idx: int
            ) -> Tensor:
        aud = self.__get_aud(aud_path)
        return self.pad_mels(aud, aud_idx)

    def __get_padded_tokens(self, text: str) -> Tensor:
        text = self.text_pipeline.run(text)
        tokens = self.tokenizer.tokens2ids(text)
        tokens.append(self.tokenizer.special_tokens.eos_id)
        tokens = self.pad_tokens(tokens)
        return torch.LongTensor(tokens)

    def prepocess_lines(self, data: str) -> List[str]:
        return [
            item.split(hprams.data.sep)
            for item in data
        ]

    def _get_max_size(self, aud_idx: int) -> int:
        """Returns the maximum length in the current batch of

        Args:
            aud_idx (int): the audio index - location in the sorted data

        Returns:
            int: the maximum size in the current batch
        """
        # if I'm at index 6 and the batch size is 8 then
        # here we still working on the first batch
        # while if I'm at index 10 and the batch size
        # is 8 then we are working on the second batch
        batch_idx = aud_idx // self.batch_size
        # if already this batch checked and
        # the max size added to max_size list
        if len(self.max_size) > batch_idx:
            return self.max_size[batch_idx]
        end = (batch_idx + 1) * self.batch_size
        start = batch_idx * self.batch_size
        if end > len(self.lengths):
            end = len(self.lengths)
        max_size_val = max(self.lengths[start: end])
        self.max_size.append(max_size_val)
        return max_size_val

    def pad_mels(self, mels: Tensor, aud_idx: int) -> Tensor:
        max_size_val = self._get_max_size(aud_idx)
        n = max_size_val - mels.shape[1] + 1
        zeros = torch.zeros(size=(1, n, mels.shape[-1]))
        return torch.cat([zeros, mels], dim=1)

    def pad_tokens(self, tokens: list) -> Tensor:
        length = self.max_len - len(tokens) + 1
        return tokens + [self.tokenizer.special_tokens.pad_id] * length
