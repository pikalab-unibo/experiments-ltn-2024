import torch

class Function:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def forward(self, *args):
        # Check the number of arguments the model's forward method expects
        num_args = self.model.forward.__code__.co_argcount - 1  # Subtract 1 for 'self'
        if num_args == 1:
            inputs = torch.cat(args, dim=-1) if len(args) > 1 else args[0]
            return self.model(inputs)
        elif num_args == len(args):
            return self.model(*args)
        else:
            raise ValueError(f"Model expects {num_args} arguments, but got {len(args)}.")

    def __call__(self, *args):
        return self.forward(*args)
