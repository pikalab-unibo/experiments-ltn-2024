from ltn_imp.parsing.expression_transformations import transform
from nltk.sem.logic import Expression
from ltn_imp.parsing.parser import LTNConverter
import torch.nn as nn

class ModuleFactory:

    def __init__(self):
        self.converter = LTNConverter()

    def get_name(self, expression):
        expression = Expression.fromstring(transform(expression))

        if hasattr(expression, 'first'):
            expression = expression.first

        while hasattr(expression, 'term'):
            expression = expression.term

        while hasattr(expression, 'function'):
            expression = expression.function

        return str(expression)

    def get_params(self, expression):

        expression = Expression.fromstring(transform(expression))

        if hasattr(expression, 'first'):
            expression = expression.first

        while hasattr(expression, 'term'):
            expression = expression.term
        
        return [str(arg) for arg in expression.args]
    
    def get_functionality(self, expression):
        return self.converter( str( Expression.fromstring( transform(expression) ).second ) ) 

    def create_module(self, expression):

        function_name = self.get_name(expression)
        params = self.get_params(expression)
        functionality = self.get_functionality(expression)
        
        # Define the forward method dynamically
        def forward(self, *args):
            local_vars = dict(zip(params, args))
            print(local_vars)
            return functionality(local_vars)

        def __call__(self, *args):
            return self.forward(*args)
        
        # Create a new class with the dynamically defined forward method
        module_class = type(function_name, (nn.Module,), {
            'forward': forward,
            '__call__': __call__
        })

        return module_class