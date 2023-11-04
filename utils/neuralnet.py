import torch.functional as F
import torch.nn as nn
import torch

class NeuralNetwork(nn.Module):
    def __init__(self, input_features, hidden_layers, output_classes, dropout = True, device = "cuda"):
        super().__init__()

        self.initialize_layers(input_features, hidden_layers, output_classes, dropout)
        self.device = device
        self.to(device)

        
    def initialize_layers(self, input_features, hidden_layers, output_classes, dropout):
        # Make it easier to create the liear layers
        dimensions = [input_features]
        dimensions.extend(hidden_layers)
        dimensions.append(output_classes)

        self.forward_layers = nn.Sequential()

        for x in range(1, len(dimensions)):
            self.forward_layers.append(
                nn.Linear(dimensions[x - 1], dimensions[x]),
            )

            if x < len(dimensions) - 1: 
                self.forward_layers.append(nn.ReLU())

                if dropout: 
                    self.forward_layers.append(nn.Dropout())

        # Note we need this so that all the layers in the list 
        # are registered as params. See here
        # https://discuss.pytorch.org/t/register-layers-within-list-as-parameters/150761/4 
        self.input_layer = self.forward_layers[0]


    def forward(self, x):
        x = x.to(self.device)
        y =  self.forward_layers(x)
        x = x.to("cpu")
        return y