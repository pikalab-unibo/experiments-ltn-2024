from abc import ABC, abstractmethod
import torch 

class Connective(ABC):
    @abstractmethod
    def forward(self, a, b):
        pass

    def __call__(self, a, b):
        return self.forward(a, b)

class AndConnective(Connective):
    def forward(self, a, b):
        # TODO:  Implement the logic for the AND operation
        return a * b

class OrConnective(Connective):
    def forward(self, a, b):
        # TODO:  Implement the logic for the OR operation
        return a + b - a * b

class ImpliesConnective(Connective):
    def forward(self, a, b):
        # TODO:  Implement the logic for the IMPLIES operation
        return torch.max(torch.tensor(1) - a, b)

class IffConnective(Connective):
    def forward(self, a, b):
        # TODO:  Implement the logic for the IFF operation (bi-conditional)
        return 1 - torch.abs(a - b)


class UnaryOperation(ABC):
    @abstractmethod
    def forward(self, a):
        pass

    def __call__(self, a):
        return self.forward(a)

class NotOperation(UnaryOperation):
    def forward(self, a):
        # TODO:  Implement the logic for the NOT operation
        return 1 - a
