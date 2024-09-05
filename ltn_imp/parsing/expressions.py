def collect_variables(expr):
    variables = set()

    if isinstance(expr, VariableExpression):
        variables.add(expr)
    elif isinstance(expr, BinaryExpression):
        variables.update(collect_variables(expr.left))
        variables.update(collect_variables(expr.right))
    elif isinstance(expr, NegatedExpression):
        variables.update(collect_variables(expr.term))
    elif isinstance(expr, NegativeExpression):
        variables.update(collect_variables(expr.term))
    elif isinstance(expr, QuantifiedExpression):
        for variable in expr.variables_:
            variables.add(variable)
        variables.update(collect_variables(expr.term))
    elif isinstance(expr, ApplicationExpression):
        for arg in expr.args:
            variables.update(collect_variables(arg))
    elif isinstance(expr, IndexExpression):
        variables.update(collect_variables(expr.variable))
        variables.update(collect_variables(expr.feature))

    return variables

class Expression:
    def variables(self):
        return collect_variables(self)
    
    
class BinaryExpression(Expression):
    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right

    def __repr__(self):
        return f"<{self.__class__.__name__}>: ({self.left} {self.operator} {self.right})"
    
    def __str__(self):
        return f"({self.left} {self.operator} {self.right})"

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
    
    def __str__(self):
        return f"~({self.term})"
    
class NegativeExpression(Expression):
    def __init__(self, term):
        self.term = term

    def __repr__(self):
        return f"<{self.__class__.__name__}>: -({self.term})"
    
    def __str__(self):
        return f"-({self.term})"

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
    def __init__(self, quantifier, variables, term):
        self.quantifier = quantifier
        self.variables_ = variables if isinstance(variables, list) else [variables]
        self.term = term

    def __repr__(self):
        temp = [ str(variable) for variable in self.variables_]
        return f"<{self.__class__.__name__}>: {self.quantifier} {temp}. ({self.term})"
    
    def __str__(self):
        temp = [ str(variable) for variable in self.variables_]
        return f"{self.quantifier} {temp}. ({self.term})"

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
    
    def __str__(self):
        args = ", ".join(map(str, self.args))
        return f"{self.function}({args})"
class VariableExpression(Expression):
    def __init__(self, variable):
        self.variable = variable

    def __repr__(self):
        return f"<{self.__class__.__name__}>: {self.variable}"
    
    def __str__(self):
        return self.variable
    
    def __eq__(self, value: object) -> bool:
        if isinstance(value, VariableExpression):
            return self.variable == value.variable
        return False

    def __hash__(self) -> int:
        return hash(self.variable)

class ConstantExpression(Expression):
    def __init__(self, constant):
        self.constant = constant

    def __repr__(self):
        return self.constant
    
class IndexExpression(Expression):
    def __init__(self, variable, feature):
        self.variable = variable
        self.feature = feature

    def __repr__(self):
        return f"<{self.__class__.__name__}>: {self.variable}[{self.feature}]"
    
    def __str__(self):
        return f"{self.variable}[{self.feature}]"