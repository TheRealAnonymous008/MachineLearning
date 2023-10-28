import pandas as pd

class CharacterTok: 
    def __init__(self):
        pass
    
    def encode(self, text : pd.Series):
        max_length = text.str.len().max()
        
        # Apply tokenization and padding
        tokenized_series = text.apply(lambda text: list(text.ljust(max_length, " ")))
        
        # Create a DataFrame from the tokenized Series
        token_matrix = pd.DataFrame(list(tokenized_series))
        
        return token_matrix
