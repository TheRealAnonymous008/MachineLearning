import torch.functional as F
import torch.nn as nn
import torch

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        # Embedding layer to convert input data into d_model-dimensional vectors
        self.embedding = nn.Linear(input_size, d_model)

        # Transformer Encoder layers
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout
            ),
            num_layers=num_layers
        )

    def forward(self, x):
        x = x.permute(1, 0, 2) 
        embedded = self.embedding(x)
        encoded = self.encoder(embedded)
        encoded = encoded.permute(1, 0, 2) 
        return encoded