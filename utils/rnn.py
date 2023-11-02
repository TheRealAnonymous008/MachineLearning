import torch.functional as F
import torch.nn as nn
import torch

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob = 0.5, device = "cuda"):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = device
        self.to(device)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Apply dropout
        out = self.dropout(out)

        # Index hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out