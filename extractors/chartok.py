import pandas as pd

class CharacterTok: 
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
