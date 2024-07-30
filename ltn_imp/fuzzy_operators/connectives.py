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

# And Connective and its subclasses
class AndConnective(BinaryConnective):
    def __init__(self, implementation, stable=True):
        self.stable = stable
        super().__init__(implementation)

class MinAndConnective(AndConnective):
    def __init__(self):
        super().__init__(self.implementation)

    def implementation(self, a, b):
        return torch.minimum(a, b)

class ProdAndConnective(AndConnective):
    def __init__(self, stable=True):
        super().__init__(self.implementation, stable=stable)

    def implementation(self, a, b):
        eps = 1e-4
        if self.stable:
            a = (1 - eps) * a + eps
            b = (1 - eps) * b + eps
        return torch.mul(a, b)

class LukAndConnective(AndConnective):
    def __init__(self):
        super().__init__(self.implementation)

    def implementation(self, a, b):
        return torch.maximum(a + b - 1, torch.zeros_like(a))

class DefaultAndConnective(MinAndConnective):
    pass

# Or Connective and its subclasses
class OrConnective(BinaryConnective):
    def __init__(self, implementation, stable=True):
        self.stable = stable
        super().__init__(implementation)

class MaxOrConnective(OrConnective):
    def __init__(self):
        super().__init__(self.implementation)

    def implementation(self, a, b):
        return torch.maximum(a, b)

class ProbSumOrConnective(OrConnective):
    def __init__(self, stable=True):
        super().__init__(self.implementation, stable=stable)

    def implementation(self, a, b):
        eps = 1e-4
        if self.stable:
            a = (1 - eps) * a
            b = (1 - eps) * b
        return a + b - a * b

class LukOrConnective(OrConnective):
    def __init__(self):
        super().__init__(self.implementation)

    def implementation(self, a, b):
        return torch.minimum(a + b, torch.ones_like(a))

class DefaultOrConnective(MaxOrConnective):
    pass

# Implies Connective and its subclasses
class ImpliesConnective(BinaryConnective):
    def __init__(self, implementation, stable=True):
        self.stable = stable
        super().__init__(implementation)

class KleeneDienesImpliesConnective(ImpliesConnective):
    def __init__(self):
        super().__init__(self.implementation)

    def implementation(self, a, b):
        # Ensure a and b have the same shape and are at least 2D tensors
        if a.ndim == 1:
            a = a.unsqueeze(1)
        if b.ndim == 1:
            b = b.unsqueeze(1)

        # Perform the operation
        result = torch.maximum(1. - a, b)
        return result

class GodelImpliesConnective(ImpliesConnective):
    def __init__(self):
        super().__init__(self.implementation)

    def implementation(self, a, b):
        return torch.where(a <= b, torch.ones_like(a), b)

class ReichenbachImpliesConnective(ImpliesConnective):
    def __init__(self, stable=True):
        super().__init__(self.implementation, stable=stable)

    def implementation(self, a, b):
        eps = 1e-4
        if self.stable:
            a = (1 - eps) * a + eps
            b = (1 - eps) * b
        return 1. - a + a * b

class GoguenImpliesConnective(ImpliesConnective):
    def __init__(self, stable=True):
        super().__init__(self.implementation, stable=stable)

    def implementation(self, a, b):
        eps = 1e-4
        if self.stable:
            a = (1 - eps) * a + eps
        return torch.where(a <= b, torch.ones_like(a), b / a)

class LukImpliesConnective(ImpliesConnective):
    def __init__(self):
        super().__init__(self.implementation)

    def implementation(self, a, b):
        return torch.minimum(1.0 - a + b, torch.ones_like(a))

class DefaultImpliesConnective(KleeneDienesImpliesConnective):
    pass

# Iff Connective and its subclasses
class IffConnective(BinaryConnective):
    def __init__(self, implementation):
        super().__init__(implementation)

class DefaultIffConnective(IffConnective):
    def __init__(self):
        super().__init__(self.implementation)

    def implementation(self, a, b):
        return 1 - torch.abs(a - b)

# Not Connective and its subclasses
class NotConnective(UnaryConnective):
    def __init__(self, implementation):
        super().__init__(implementation)

class StandardNotConnective(NotConnective):
    def __init__(self):
        super().__init__(self.implementation)

    def implementation(self, a):
        return 1 - a

class GodelNotConnective(NotConnective):
    def __init__(self):
        super().__init__(self.implementation)

    def implementation(self, a):
        return torch.eq(a, 0.).float()

class DefaultNotConnective(StandardNotConnective):
    pass

class EqConnective(BinaryConnective):
    def __init__(self, implementation):
        super().__init__(implementation)

class DefaultEqConnective(EqConnective):
    def __init__(self):
        super().__init__(self.implementation)

    def implementation(self, a, b):
        # If 'a' and 'b' have the same shape, compare elementwise
        return  1 - torch.tanh(torch.abs(a - b))**2