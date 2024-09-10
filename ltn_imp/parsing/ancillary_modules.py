import torch.nn as nn
import inspect

class ModuleFactory:

    def __init__(self, converter):
        self.converter = converter
    
    def get_functionality(self, expression):
        return self.converter(expression)

    def create_module(self, name, params, functionality):
        functionality = self.get_functionality(functionality)
        
        # Define the forward method dynamically
        def forward(self, *args):
            local_vars = dict(zip(params, args))
            return functionality(local_vars)

        # Create a signature with the given parameters
        new_params = [inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + \
                     [inspect.Parameter(param, inspect.Parameter.POSITIONAL_OR_KEYWORD) for param in params]
        
        new_sig = inspect.Signature(parameters=new_params)
        
        forward.__signature__ = new_sig

        def __call__(self, *args):
            return self.forward(*args)
        
        # Create a new class with the dynamically defined forward method
        module_class = type(str(name), (nn.Module,), {
            'forward': forward,
            '__call__': __call__
        })
        
        self.converter.predicates[name] = module_class()

        return module_class