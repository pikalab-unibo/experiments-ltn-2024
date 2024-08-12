import tatsu
from ltn_imp.parsing.expressions import *


class FOLSemantics:
    def start(self, ast):
        return ast
    
    def expression(self, ast):
        return ast
    
    def equivalence(self, ast):
        return IffExpression(ast[0], ast[2])
    
    def implication(self, ast):
        return ImpExpression(ast[0], ast[2])
    
    def disjunction(self, ast):
        return OrExpression(ast[0], ast[2])
    
    def conjunction(self, ast):
        return AndExpression(ast[0], ast[2])
    
    def negation(self, ast):
        return NegatedExpression(ast[1])
    
    def quantified(self, ast):
        quantifier = ExistsExpression if ast[0] == 'exists' else ForallExpression
        return quantifier(ast[1], ast[3])
    
    def predicate(self, ast):
        name = ast[0]  # The name of the predicate
        variables = ast[1]  # The variable within the predicate
        if type(variables) != list:
            variables = [variables]
        return ApplicationExpression(name, variables)
    
    def variable(self, ast):
        return VariableExpression(variable=ast)
    
    def constant(self, ast):
        return ConstantExpression(constant=ast)
    
    def addition(self, ast):
        return AdditionExpression(ast[0], ast[2])
    
    def subtraction(self, ast):
        return SubtractionExpression(ast[0], ast[2])
    
    def multiplication(self, ast):
        return MultiplicationExpression(ast[0], ast[2])
    
    def division(self, ast):
        return DivisionExpression(ast[0], ast[2])
    
    def less_than(self, ast):
        return LessThanExpression(ast[0], ast[2])
    
    def more_than(self, ast):
        return MoreThanExpression(ast[0], ast[2])
    
    def less_equal(self, ast):
        return LessEqualExpression(ast[0], ast[2])
    
    def more_equal(self, ast):
        return MoreEqualExpression(ast[0], ast[2])

    def equality(self, ast):
        return EqualityExpression(ast[0], ast[2])
    
    def direct(self, ast):
        return DirectEqualityExpression(ast[0], ast[2])
    
    def negative(self, ast):
        return NegativeExpression(ast[1])
    
    def index(self, ast):
        return IndexExpression(ast[0], ast[1])

class LTNParser():
    def __init__(self, path):
        with open(path, 'r') as file:
            grammar = file.read()

        self.parser = tatsu.compile(grammar)

    def parse(self, text):
        return self.parser.parse(text, semantics=FOLSemantics())

    def __call__(self, text):
        return self.parse(text)