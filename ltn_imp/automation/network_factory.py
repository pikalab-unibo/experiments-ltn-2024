import torch.nn as nn

class NNFactory:
    def __init__(self):
        pass
    
    def __call__(self, layers, activations, regularizations):
        network_layers = []

        for (in_size, out_size), activation, regularization_list in zip(layers, activations, regularizations):
            # Add linear layer
            linear_layer = nn.Linear(in_size, out_size)
            network_layers.append(linear_layer)
            
            # Apply regularizations from the list
            if regularization_list:
                for regularization in regularization_list:
                    reg_layers = self._get_regularization(regularization, linear_layer)
                    if reg_layers is not None:
                        network_layers.extend(reg_layers)
            
            # Add activation function
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

    def _get_regularization(self, regularization, linear_layer):
        reg_layers = []
        
        if regularization is None:
            return reg_layers
        
        # Add dropout layer if specified
        if 'dropout' in regularization:
            reg_layers.append(nn.Dropout(regularization['dropout']))
        
        # Add batch normalization if specified
        if 'batch_norm' in regularization and regularization['batch_norm']:
            reg_layers.append(nn.BatchNorm1d(linear_layer.out_features))
        
        # Add layer normalization if specified
        if 'layer_norm' in regularization and regularization['layer_norm']:
            reg_layers.append(nn.LayerNorm(linear_layer.out_features))
        
        return reg_layers