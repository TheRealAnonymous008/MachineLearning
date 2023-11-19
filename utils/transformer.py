import torch.functional as F
import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=512, device = "cuda"):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        # Create positional embeddings
        position = torch.arange(0, max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        pos_embed = torch.zeros(1, max_seq_length, d_model)
        pos_embed[0, :, 0::2] = torch.sin(position * div_term)
        pos_embed[0, :, 1::2] = torch.cos(position * div_term)
        pos_embed = pos_embed.to(device)

        self.pos_embed = pos_embed
        
        self.device = device

    def forward(self, x):
        x = x + self.pos_embed
        return self.dropout(x)
    
class TransformerEncoder(nn.Module):
    def __init__(self, input_size, output_size, embed_dims, num_heads, num_layers, ff = 10, dropout=0.1, device = "cuda"):
        super().__init__()

        # Embedding layer to convert input data into d_model-dimensional vectors
        self.embedding = nn.Embedding(255, embed_dims, ord("\n"))

        # Positional encoding
        self.positional_encoding = PositionalEncoding(embed_dims, input_size)

        # Transformer Encoder layers
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dims,
                nhead=num_heads,
                dim_feedforward= ff,
                dropout=dropout
            ),
            num_layers=num_layers
        )
        
        self.fc = nn.Linear(embed_dims, output_size)
        self.device = device
        self.to(device)

    def forward(self, x, last_pad = -1):
        x = x.to(self.device)
        embedded = self.embedding(x) 
        embedded = self.positional_encoding(embedded)
        
        encoded = self.encoder.forward(embedded)
        encoded = encoded.permute(1, 0, 2) 

        y = self.fc(encoded[-1, :, :])
        x = x.to("cpu")

        return y