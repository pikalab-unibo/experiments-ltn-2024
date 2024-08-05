import torch.nn as nn
import inspect

class ModuleFactory:

    def __init__(self, converter):
        self.converter = converter

    def get_name(self, expression):
        expression = self.converter.parse(expression)
        
        while hasattr(expression, 'term'):
            expression = expression.term

        if hasattr(expression, 'first'):
            expression = expression.first
            
        while hasattr(expression, 'function'):
            expression = expression.function

        return str(expression)

    def get_params(self, expression):

        expression = self.converter.parse(expression)

        while hasattr(expression, 'term'):
            expression = expression.term
        
        if hasattr(expression, 'first'):
            expression = expression.first
        
        return [str(arg) for arg in expression.args]
    
    def get_functionality(self, expression):

        expression = self.converter.parse(expression)

        while hasattr(expression, 'term'):
            expression = expression.term

        return self.converter(str(expression.second)) 

    def create_module(self, expression):

        function_name = self.get_name(expression)
        params = self.get_params(expression)
        functionality = self.get_functionality(expression)
        
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
        module_class = type(str(function_name), (nn.Module,), {
            'forward': forward,
            '__call__': __call__
        })
        
        self.converter.predicates[function_name] = module_class()

        return module_class