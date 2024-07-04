import torch
import inspect

class Predicate:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def forward(self, *args, **kwargs):
        # Get the signature of the model's forward method
        signature = inspect.signature(self.model.forward)
        parameters = signature.parameters
        
        # Extract only the required parameters, ignoring 'self'
        required_params = [p for p in parameters.values() if p.default == inspect.Parameter.empty and p.name != 'self']

        # Check if the number of provided arguments matches the required ones
        if len(args) == len(required_params):
            return self.model(*args, **kwargs)
        else:
            # Match args to the model's parameters
            bound_args = signature.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()
            return self.model(*bound_args.args, **bound_args.kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
