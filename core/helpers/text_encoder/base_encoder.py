from abc import ABC, abstractmethod


class base_encoder(ABC):
    def __init__(self, alphabet):
        self.symbols = list(alphabet)

    @abstractmethod
    def encode(self, texts, max_len=25):
        pass

    @abstractmethod
    def decode(self, indices, lengths):
        pass

