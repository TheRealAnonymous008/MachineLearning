import torch.functional as F
import torch.nn as nn
import torch

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, output_size, embed_dims, num_heads, num_layers, dropout=0.1, device = "cuda"):
        super().__init__()

        # Embedding layer to convert input data into d_model-dimensional vectors
        self.embedding = nn.Embedding(255, embed_dims, ord("\n"))

        # Transformer Encoder layers
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dims,
                nhead=num_heads,
                dim_feedforward= 4 * embed_dims,
                dropout=dropout
            ),
            num_layers=num_layers
        )
        
        self.fc = nn.Linear(embed_dims, output_size)
        self.device = device
        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        embedded = self.embedding(x)
        encoded = self.encoder(embedded)
        encoded = encoded.permute(1, 0, 2) 

        y = self.fc(encoded[-1, :, :])

        x = x.to("cpu")
        return y