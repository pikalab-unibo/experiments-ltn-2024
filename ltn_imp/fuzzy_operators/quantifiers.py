from abc import ABC, abstractmethod
import torch 

class Quantifier(ABC):
    @abstractmethod
    def forward(self, *args):
        pass

    def __call__(self, *args):
        return self.forward(*args)

class ExistsQuantifier(Quantifier):
    # TODO:  Implement the logic for the Exists operation
    def forward(self, *args):
        # Ensure args are tensors
        tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
        return torch.max(torch.stack(tensors))

class ForallQuantifier(Quantifier):
    # TODO:  Implement the logic for the Forall operation
    def forward(self, *args):
        # Ensure args are tensors
        tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
        return torch.min(torch.stack(tensors))