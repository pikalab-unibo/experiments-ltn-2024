import torch.nn as nn

class NNFactory:
    def __init__(self):
        pass
    
    def __call__(self, layers, activations):
        network_layers = []

        for (in_size, out_size), activation in zip(layers, activations):
            network_layers.append(nn.Linear(in_size, out_size))
            if activation is not None:
                network_layers.append(self._get_activation(activation))
        
        return nn.Sequential(*network_layers)
    
    def _get_activation(self, activation):
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(),
            'softmax': nn.Softmax(dim=1),
            'identity': nn.Identity()
        }
        return activations.get(activation, nn.Identity())  # Use Identity if activation is None
