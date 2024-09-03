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

class NegativeConnective(UnaryConnective):
    def __init__(self, implementation):
        super().__init__(implementation)

class DefaultNegativeConnective(NegativeConnective):
    def __init__(self):
        super().__init__(self.implementation)

    def implementation(self, a):
        return torch.sub(0,a)

class EqConnective(BinaryConnective):
    def __init__(self, implementation):
        super().__init__(implementation)

class TanEqConnective(EqConnective): # This returns a tensor of the form [ [0], [1], [0], [1], [0], [1], ... ]
    def __init__(self):
        super().__init__(self.implementation)

    def implementation(self, a, b):
        # Ensure a and b have the same shape and are at least 2D tensors
        if a.ndim == 1:
            a = a.unsqueeze(1)
        if b.ndim == 1:
            b = b.unsqueeze(1)

        # Perform the operation
        result = 1 - torch.tanh(2 * torch.abs(torch.sub(a,b)))**2        
        return result
    
class SqrtEqConnective(EqConnective):  # This returns a tensor in the form [ 0, 1, 0, 1, 0, 1, ...]
    def __init__(self):
        super().__init__(self.implementation)

    def implementation(self, a, b):
        
        # Ensure a and b have the samzaze shape and are at least 2D tensors
        if a.ndim == 0:
            a = a.unsqueeze(0)
        if b.ndim == 0:
            b = b.unsqueeze(0)
        if a.ndim == 1:
            a = a.unsqueeze(1)
        if b.ndim == 1:
            b = b.unsqueeze(1)

        alpha = 0.005  # You can adjust the value of alpha as needed
        result = torch.exp(-alpha * torch.sqrt(torch.sum(torch.square(a - b), dim=1)))

        return result
    
DefaultEqConnective = SqrtEqConnective

# LessThan Connective
class LessThanConnective(BinaryConnective):
    def __init__(self, k=10):
        self.k = k
        super().__init__(self.implementation)

    def normalize_min_max(self, tensor1, tensor2):
        
        if tensor2.dim() == 0:
            tensor2 = tensor2.unsqueeze(0)

        combined_min = torch.min(torch.cat((tensor1, tensor2)))
        combined_max = torch.max(torch.cat((tensor1, tensor2)))
        tensor1_normalized = (tensor1 - combined_min) / (combined_max - combined_min)
        tensor2_normalized = (tensor2 - combined_min) / (combined_max - combined_min)
        return tensor1_normalized, tensor2_normalized

    def implementation(self, tensor1, tensor2):
        tensor1_normalized, tensor2_normalized = self.normalize_min_max(tensor1, tensor2)
        result = torch.sigmoid(self.k * (tensor2_normalized - tensor1_normalized))
        return result

    
class DefaultLessThanConnective(LessThanConnective):
    pass

# MoreThan Connective
class MoreThanConnective(BinaryConnective):
    def __init__(self, k=10):
        self.k = k
        super().__init__(self.implementation)

    def normalize_min_max(self, tensor1, tensor2):
        
        if tensor2.dim() == 0:
            tensor2 = tensor2.unsqueeze(0)

        combined_min = torch.min(torch.cat((tensor1, tensor2)))
        combined_max = torch.max(torch.cat((tensor1, tensor2)))
        tensor1_normalized = (tensor1 - combined_min) / (combined_max - combined_min)
        tensor2_normalized = (tensor2 - combined_min) / (combined_max - combined_min)
        return tensor1_normalized, tensor2_normalized

    def implementation(self, tensor1, tensor2):
        tensor1_normalized, tensor2_normalized = self.normalize_min_max(tensor1, tensor2)
        result = torch.sigmoid(self.k * (tensor1_normalized - tensor2_normalized))
        return result


class DefaultMoreThanConnective(MoreThanConnective):
 pass

# Add Connective
class AddConnective(BinaryConnective):
    def __init__(self):
        super().__init__(self.implementation)

    def implementation(self, tensor1, tensor2):
        return torch.add(tensor1, tensor2).float()

class DefaultAddConnective(AddConnective):
 pass

# Subtract Connective
class SubtractConnective(BinaryConnective):
    def __init__(self):
        super().__init__(self.implementation)

    def implementation(self, tensor1, tensor2):
        return torch.sub(tensor1, tensor2).float()

class DefaultSubtractConnective(SubtractConnective):
    pass

# Multiply Connective
class MultiplyConnective(BinaryConnective):
    def __init__(self):
        super().__init__(self.implementation)

    def implementation(self, tensor1, tensor2):
        return torch.mul(tensor1, tensor2).float()

class DefaultMultiplyConnective(MultiplyConnective):
    pass
# Divide Connective
class DivideConnective(BinaryConnective):
    def __init__(self):
        super().__init__(self.implementation)

    def implementation(self, tensor1, tensor2):
        return torch.div(tensor1, tensor2).float()

class DefaultDivideConnective(DivideConnective):
    pass


class LessThanOrEqualConnective(BinaryConnective):
    def __init__(self, k=5):
        self.k = k
        super().__init__(self.implementation)

    def implementation(self, tensor1, tensor2):
        return DefaultOrConnective()(
            DefaultLessThanConnective(self.k)(tensor1, tensor2),
            DefaultEqConnective()(tensor1, tensor2)
        )

class DefaultLessThanOrEqualConnective(LessThanOrEqualConnective):
    pass

# MoreThanOrEqual Connective
class MoreThanOrEqualConnective(BinaryConnective):
    def __init__(self, k=5):
        self.k = k
        super().__init__(self.implementation)

    def implementation(self, tensor1, tensor2):
        return DefaultOrConnective()(
            DefaultLessThanConnective(self.k)(tensor2, tensor1),
            DefaultEqConnective()(tensor1, tensor2)
        )

class DefaultMoreThanOrEqualConnective(MoreThanOrEqualConnective):
    pass