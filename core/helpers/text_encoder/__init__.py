from .ctc_encoder import ctc_encoder


class text_encoder:
    def __init__(self, kind, alphabet):
        if kind == "CTC":
            self._encoder = ctc_encoder(alphabet)
        else:
            raise ValueError(f"Unknown encoder kind: {kind}")

    def encode(self, texts, max_len=25):
        return self._encoder.encode(texts, max_len)

    def decode(self, indices, lengths):
        return self._encoder.decode(indices, lengths)

    @property
    def symbols(self):
        return self._encoder.symbols

