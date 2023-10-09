import torch.functional as F
import torch.nn as nn
import torch

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMNetwork, self).__init__()
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Define a fully connected layer for output
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Forward pass through the LSTM layer
        out, (hn, cn) = self.lstm(x)
        
        # Select the last time step's output
        out = out[:, -1, :]
        
        # Forward pass through the fully connected layer
        out = self.fc(out)
        return out