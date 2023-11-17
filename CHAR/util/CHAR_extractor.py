import pandas as pd
from torch.utils.data import Dataset
import torch

class CharacterTok: 
    pad_token = " "
    def __init__(self):
        pass
    
    def encode(self, text : pd.Series):
        max_length = text.str.len().max()
        
        # Apply tokenization and padding
        char_to_int = {char: i for i, char in enumerate(' abcdefghijklmnopqrstuvwxyz')}
        tokenized_series = text.apply(lambda text: [char_to_int[char] for char in text.ljust(max_length, self.pad_token)])
        
        
        # Create a DataFrame from the tokenized Series
        token_matrix = pd.DataFrame(list(tokenized_series))
        
        return token_matrix

class CharTokenDataset(Dataset):
    def __init__(self, data, labels, tokenizer, max_seq_length, normalize = True, dtype = torch.float32):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.normalize = normalize
        self.dtype = dtype

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data['text'].iloc[idx]
        # Tokenize the text by characters
        tokens = list(text)
        # Pad or truncate the tokens to the specified max_seq_length
        if len(tokens) < self.max_seq_length:
            tokens += ["\n"] * (self.max_seq_length - len(tokens))
        else:
            tokens = tokens[:self.max_seq_length]
        # Convert tokens to numerical representations (e.g., ASCII values).  
        c = 1.0 
        if self.normalize:
            c = 255.0

        token_ids = [ord(char) / c  for char in tokens]

        label = self.labels.iloc[idx]  
        return torch.tensor(token_ids, dtype=self.dtype), label
    

