class Expression:
    pass

class BinaryExpression(Expression):
    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right

    def __repr__(self):
        return f"<{self.__class__.__name__}>: ({self.left} {self.operator} {self.right})"

class AndExpression(BinaryExpression):
    def __init__(self, left, right):
        super().__init__(left, '&', right)

class OrExpression(BinaryExpression):
    def __init__(self, left, right):
        super().__init__(left, '|', right)

class ImpExpression(BinaryExpression):
    def __init__(self, left, right):
        super().__init__(left, '->', right)

class IffExpression(BinaryExpression):
    def __init__(self, left, right):
        super().__init__(left, '<->', right)

class NegatedExpression(Expression):
    def __init__(self, term):
        self.term = term

    def __repr__(self):
        return f"<{self.__class__.__name__}>: ~({self.term})"

class EqualityExpression(BinaryExpression):
    def __init__(self, left, right):
        super().__init__(left, '=', right)

class  DirectEqualityExpression(BinaryExpression):
    def __init__(self, left, right):
        super().__init__(left, '==', right)

class AdditionExpression(BinaryExpression):
    def __init__(self, left, right):
        super().__init__(left, '+', right)

class SubtractionExpression(BinaryExpression):
    def __init__(self, left, right):
        super().__init__(left, '-', right)

class MultiplicationExpression(BinaryExpression):
    def __init__(self, left, right):
        super().__init__(left, '*', right)

class DivisionExpression(BinaryExpression):
    def __init__(self, left, right):
        super().__init__(left, '/', right)

class LessThanExpression(BinaryExpression):
    def __init__(self, left, right):
        super().__init__(left, '<', right)

class MoreThanExpression(BinaryExpression):
    def __init__(self, left, right):
        super().__init__(left, '>', right)

class LessEqualExpression(BinaryExpression):
    def __init__(self, left, right):
        super().__init__(left, '<=', right)

class MoreEqualExpression(BinaryExpression):
    def __init__(self, left, right):
        super().__init__(left, '>=', right)

class QuantifiedExpression(Expression):
    def __init__(self, quantifier, variable, term):
        self.quantifier = quantifier
        self.variable = variable
        self.term = term

    def __repr__(self):
        return f"<{self.__class__.__name__}>: {self.quantifier} {self.variable}.({self.term})"

class ExistsExpression(QuantifiedExpression):
    def __init__(self, variable, term):
        super().__init__('∃', variable, term)

class ForallExpression(QuantifiedExpression):
    def __init__(self, variable, term):
        super().__init__('∀', variable, term)

class ApplicationExpression(Expression):
    def __init__(self, function, args):
        self.function = function  # The function or predicate being applied
        self.args = args

    def __repr__(self):
        args = ", ".join(map(str, self.args))
        return f"<{self.__class__.__name__}>: {self.function}({args})"
    
class VariableExpression(Expression):
    def __init__(self, variable):
        self.variable = variable

    def __repr__(self):
        return f"<{self.__class__.__name__}>: {self.variable}"
    
    def __str__(self):
        return self.variable

class ConstantExpression(Expression):
    def __init__(self, constant):
        self.constant = constant

    def __repr__(self):
        return self.constant