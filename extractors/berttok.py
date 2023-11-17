

from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd 
import numpy as np 

class BertTokenizer: 
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    
    def __init__(self):
        pass

    def encode(self, sentences : pd.Series): 
        encoded_outputs =  []
        for sentence in sentences: 
            inputs = self.tokenizer.encode(sentence, return_tensors="pt")
            outputs = self.model(inputs)
            output = outputs.logits[:, -1, :].view(-1)

            encoded_outputs.append(output.detach().numpy())
            del inputs

        return np.array(encoded_outputs)
