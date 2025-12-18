import torch.nn as nn


class bidirectional_rnn(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(bidirectional_rnn, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.proj = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        try:
            self.lstm.flatten_parameters()
        except:
            pass
        out, _ = self.lstm(x)
        return self.proj(out)

