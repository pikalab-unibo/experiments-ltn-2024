from abc import ABC, abstractmethod
import torch

# Base class for all connectives
class Connective(ABC):
    @abstractmethod
    def forward(self, *args):
        pass

    def __call__(self, *args):
        return self.forward(*args)

# Base class for binary connectives
class BinaryConnective(Connective):
    def __init__(self, implementation):
        self.implementation = implementation

    def forward(self, a, b):
        return self.implementation(a, b)

# Base class for unary operations
class UnaryConnective(Connective):
    def __init__(self, implementation):
        self.implementation = implementation

    def forward(self, a):
        return self.implementation(a)

class AndConnective(BinaryConnective):
    def __init__(self, implementation_name="min", stable=True):
        self.stable = stable
        method_name = f"and_{implementation_name}"

        if not hasattr(self, method_name):
            raise ValueError(f"Unknown implementation: {implementation_name}")

        implementation_method = getattr(self, method_name)
        super().__init__(implementation_method)

    def and_min(self, a, b):
        return torch.minimum(a, b)

    def and_prod(self, a, b):
        eps = 1e-4
        if self.stable:
            a = (1 - eps) * a + eps
            b = (1 - eps) * b + eps
        return torch.mul(a, b)

    def and_luk(self, a, b):
        return torch.maximum(a + b - 1, torch.zeros_like(a))

class OrConnective(BinaryConnective):
    def __init__(self, implementation_name="max", stable=True):
        self.stable = stable
        method_name = f"or_{implementation_name}"

        if not hasattr(self, method_name):
            raise ValueError(f"Unknown implementation: {implementation_name}")

        implementation_method = getattr(self, method_name)
        super().__init__(implementation_method)

    def or_max(self, a, b):
        return torch.maximum(a, b)

    def or_prob_sum(self, a, b):
        eps = 1e-4
        if self.stable:
            a = (1 - eps) * a
            b = (1 - eps) * b
        return a + b - a * b

    def or_luk(self, a, b):
        return torch.minimum(a + b, torch.ones_like(a))

class ImpliesConnective(BinaryConnective):
    def __init__(self, implementation_name="kleene_dienes", stable=True):
        self.stable = stable
        method_name = f"implies_{implementation_name}"

        if not hasattr(self, method_name):
            raise ValueError(f"Unknown implementation: {implementation_name}")

        implementation_method = getattr(self, method_name)
        super().__init__(implementation_method)

    def implies_kleene_dienes(self, a, b):
        return torch.maximum(1. - a, b)

    def implies_godel(self, a, b):
        return torch.where(a <= b, torch.ones_like(a), b)

    def implies_reichenbach(self, a, b):
        eps = 1e-4
        if self.stable:
            a = (1 - eps) * a + eps
            b = (1 - eps) * b
        return 1. - a + a * b

    def implies_goguen(self, a, b):
        eps = 1e-4
        if self.stable:
            a = (1 - eps) * a + eps
        return torch.where(a <= b, torch.ones_like(a), b / a)

    def implies_luk(self, a, b):
        return torch.minimum(1.0 - a + b, torch.ones_like(a))

class IffConnective(BinaryConnective):
    def __init__(self, implementation_name="default"):
        method_name = f"iff_{implementation_name}"

        if not hasattr(self, method_name):
            raise ValueError(f"Unknown implementation: {implementation_name}")

        implementation_method = getattr(self, method_name)
        super().__init__(implementation_method)

    def iff_default(self, a, b):
        return 1 - torch.abs(a - b)

class NotConnective(UnaryConnective):
    def __init__(self, implementation_name="standard"):
        method_name = f"not_{implementation_name}"

        if not hasattr(self, method_name):
            raise ValueError(f"Unknown implementation: {implementation_name}")

        implementation_method = getattr(self, method_name)
        super().__init__(implementation_method)

    def not_standard(self, a):
        return 1 - a

    def not_godel(self, a):
        return torch.eq(a, 0.).float()
