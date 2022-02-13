from pathlib import Path
from model import Model
from tokenizer import CharTokenizer, ITokenizer
from utils import get_formated_date, load_stat_dict
from torch.optim import Optimizer
from data import AudioPipeline, DataLoader, TextPipeline
from typing import Callable, Union
from torch.nn import Module
from functools import wraps
from hprams import hprams
from tqdm import tqdm
import torch
import os

OPT = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.Adam
}


def save_checkpoint(func) -> Callable:
    """Save a checkpoint after each iteration
    """
    @wraps(func)
    def wrapper(obj, *args, **kwargs):
        result = func(obj, *args, **kwargs)
        if not os.path.exists(hprams.training.checkpoints_dir):
            os.mkdir(hprams.training.checkpoints_dir)
        timestamp = get_formated_date()
        model_path = os.path.join(
            hprams.training.checkpoints_dir,
            timestamp + '.pt'
            )
        torch.save(obj.model.state_dict(), model_path)
        print(f'checkpoint saved to {model_path}')
        return result
    return wrapper


class Trainer:
    __train_loss_key = 'train_loss'
    __test_loss_key = 'test_loss'

    def __init__(
            self,
            criterion: Module,
            optimizer: Optimizer,
            model: Module,
            device: str,
            train_loader: DataLoader,
            test_loader: DataLoader,
            sos_token_id: int,
            epochs: int
            ) -> None:
        self.criterion = criterion
        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.step_history = dict()
        self.history = dict()
        self.sos_token_id = sos_token_id

    def fit(self):
        """The main training loop that train the model on the training
        data then test it on the test set and then log the results
        """
        for _ in range(self.epochs):
            self.train()
            self.test()
            self.print_results()

    def set_train_mode(self) -> None:
        """Set the models on the training mood
        """
        self.model = self.model.train()

    def set_test_mode(self) -> None:
        """Set the models on the testing mood
        """
        self.model = self.model.eval()

    def print_results(self):
        """Prints the results after each epoch
        """
        result = ''
        for key, value in self.history.items():
            result += f'{key}: {str(value[-1])}, '
        print(result[:-2])

    def test(self):
        """Iterate over the whole test data and test the models
        for a single epoch
        """
        total_loss = 0
        self.set_test_mode()
        for x, y in tqdm(self.test_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            max_len = y.shape[1]
            x = torch.squeeze(x, dim=1)
            result = self.model(
                x, self.sos_token_id,
                max_len, y,
                hprams.training.p_teacher_forcing
                )
            result = result.reshape(-1, result.shape[-1])
            y = y.reshape(-1)
            y = torch.squeeze(y)
            loss = self.criterion(torch.squeeze(result), y)
            total_loss += loss.item()
        total_loss /= len(self.test_loader)
        if self.__test_loss_key in self.history:
            self.history[self.__test_loss_key].append(total_loss)
        else:
            self.history[self.__test_loss_key] = [total_loss]

    @save_checkpoint
    def train(self):
        """Iterates over the whole training data and train the models
        for a single epoch
        """
        total_loss = 0
        self.set_train_mode()
        for x, y in tqdm(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            max_len = y.shape[1]
            x = torch.squeeze(x, dim=1)
            self.optimizer.zero_grad()
            result = self.model(
                x, self.sos_token_id,
                max_len, y,
                hprams.training.p_teacher_forcing
                )
            result = result.reshape(-1, result.shape[-1])
            y = y.reshape(-1)
            y = torch.squeeze(y)
            loss = self.criterion(torch.squeeze(result), y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        total_loss /= len(self.train_loader)
        if self.__train_loss_key in self.history:
            self.history[self.__train_loss_key].append(total_loss)
        else:
            self.history[self.__train_loss_key] = [total_loss]


def get_model_args(vocab_size: int) -> dict:
    device = hprams.device
    enc_params = dict(**hprams.model.encoder)
    dec_params = dict(
        **hprams.model.decoder,
        vocab_size=vocab_size
        )
    att_params = hprams.model.attention
    return {
        'enc_params': enc_params,
        'dec_params': dec_params,
        'att_params': att_params,
        'device': device
    }


def load_model(vocab_size: int) -> Module:
    model = Model(**get_model_args(vocab_size))
    if hprams.checkpoint is not None:
        load_stat_dict(model, hprams.checkpoint)
    return model

def get_tokenizer():
    tokenizer = CharTokenizer()
    if hprams.tokenizer.tokenizer_file is not None:
        tokenizer = tokenizer.load_tokenizer(
            hprams.tokenizer.tokenizer_file
            )
    tokenizer = tokenizer.add_pad_token().add_sos_token().add_eos_token()
    with open(hprams.tokenizer.vocab_path, 'r') as f:
        vocab = f.read().split('\n')
    tokenizer.set_tokenizer(vocab)
    tokenizer.save_tokenizer('tokenizer.json')
    return tokenizer
    
def get_data_loader(
        file_path: Union[str, Path],
        tokenizer: ITokenizer
        ):
    audio_pipeline = AudioPipeline()
    text_pipeline = TextPipeline()
    return DataLoader(
        file_path,
        text_pipeline,
        audio_pipeline,
        tokenizer,
        hprams.training.batch_size,
        hprams.data.max_str_len
    )
    
def get_trainer():
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    train_loader = get_data_loader(
        hprams.data.training_file,
        tokenizer
    )
    test_loader = get_data_loader(
        hprams.data.testing_file,
        tokenizer
    )
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=tokenizer.special_tokens.pad_id
        )
    model = load_model(vocab_size)
    optimizer = OPT[hprams.training.optimizer](
        model.parameters(),
        lr=hprams.training.learning_rate
        )
    return Trainer(
        criterion=criterion,
        optimizer=optimizer,
        model=model,
        device=hprams.device,
        train_loader=train_loader,
        test_loader=test_loader,
        sos_token_id=tokenizer.special_tokens.sos_id,
        epochs=hprams.training.epochs
    )

if __name__ == '__main__':
    trainer = get_trainer()
    trainer.fit()