from .base_encoder import base_encoder
import torch


class ctc_encoder(base_encoder):
    def __init__(self, alphabet):
        super().__init__(alphabet)
        self.char_to_idx = {}
        for i, ch in enumerate(alphabet):
            self.char_to_idx[ch] = i + 1
        self.symbols = ["[blank]"] + self.symbols
        self.blank_idx = 0

    def encode(self, texts, max_len=25):
        lengths = [len(t) for t in texts]
        joined = "".join(texts)
        indices = [self.char_to_idx[ch] for ch in joined]
        return torch.IntTensor(indices), torch.IntTensor(lengths)

    def decode(self, indices, lengths):
        results = []
        pos = 0
        for length in lengths:
            seq = indices[pos:pos + length]
            chars = []
            for i in range(length):
                if seq[i] != self.blank_idx and (i == 0 or seq[i - 1] != seq[i]):
                    chars.append(self.symbols[seq[i]])
            results.append(''.join(chars))
            pos += length
        return results

