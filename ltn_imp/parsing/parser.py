from nltk.sem import logic
from ltn_imp.fuzzy_operators.connectives import AndConnective, OrConnective, NotConnective, ImpliesConnective, IffConnective
from ltn_imp.fuzzy_operators.predicates import Predicate
from ltn_imp.fuzzy_operators.quantifiers import ForallQuantifier, ExistsQuantifier
from ltn_imp.fuzzy_operators.functions import Function
from ltn_imp.visitor import Visitor, make_visitable
from nltk.sem.logic import Expression
import ltn_imp.fuzzy_operators.connectives as Connectives

make_visitable(logic.Expression)


def get_subclass_with_prefix(module, superclass: type, prefix: str = "default"):
    prefix = prefix.lower()
    for k in dir(module):
        obj = getattr(module, k)
        if isinstance(obj, type) and k.lower().startswith(prefix) and issubclass(obj, superclass):
            return obj()
    
    raise KeyError(f'No subtype of {superclass} found in module {module} with prefix "{prefix}"')


class ExpressionVisitor(Visitor):

    def __init__(self, predicates, functions, connective_impls=None, quantifier_impls=None):
        connective_impls = connective_impls or {}
        quantifier_impls = quantifier_impls or {}

        self.predicates = predicates
        self.functions = functions

        And = get_subclass_with_prefix(module=Connectives, superclass=AndConnective, prefix=connective_impls.get('and', 'default'))
        Or = get_subclass_with_prefix(module=Connectives, superclass=OrConnective, prefix=connective_impls.get('or', 'default'))
        Not = get_subclass_with_prefix(module=Connectives, superclass=NotConnective, prefix=connective_impls.get('not', 'default'))
        Implies = get_subclass_with_prefix(module=Connectives, superclass=ImpliesConnective, prefix=connective_impls.get('implies', 'default'))
        Equiv = get_subclass_with_prefix(module=Connectives, superclass=IffConnective, prefix=connective_impls.get('iff', 'default'))

        Exists = ExistsQuantifier(method=quantifier_impls.get('exists', 'pmean'))
        Forall = ForallQuantifier(method=quantifier_impls.get('forall', 'min'))

        self.connective_map = {
            logic.AndExpression: And,
            logic.OrExpression: Or,
            logic.ImpExpression: Implies,
            logic.IffExpression: Equiv,
            logic.NegatedExpression : Not
        }

        self.quantifier_map = {
            logic.ExistsExpression: Exists,
            logic.AllExpression: Forall
        }

    def visit_ApplicationExpression(self, expression):
        functor = expression.function
        while hasattr(functor, 'function'):
            functor = functor.function
        functor = str(functor)
        if functor in self.predicates:
            return lambda var_mapping: Predicate(self.predicates[functor])(*[var_mapping[str(var)] for var in expression.variables()])
        elif functor in self.functions:
            return lambda var_mapping: Function(self.functions[functor])(*[var_mapping[str(var)] for var in expression.variables()])
        else:
            raise ValueError(f"Unknown functor: {type(functor)}")

    def visit_BinaryExpression(self, expression):
        connective = self.connective_map.get(type(expression))
        if connective:
            left = self.visit(expression.first)
            right = self.visit(expression.second)
            return lambda var_mapping: connective(left(var_mapping), right(var_mapping))
        else:
            raise NotImplementedError(f"Unsupported binary expression type: {type(expression)}")

    def visit_NegatedExpression(self, expression):
        connective = self.connective_map.get(type(expression))
        term = self.visit(expression.term)
        return lambda var_mapping: connective(term(var_mapping))

    def visit_QuantifiedExpression(self, expression):
        quantifier = self.quantifier_map.get(type(expression))
        if quantifier:
            term = self.visit(expression.term)
            return lambda variable_mapping: quantifier(term(variable_mapping))
        else:
            raise NotImplementedError(f"Unsupported quantifier expression type: {type(expression)}")


def convert_to_ltn(expression, predicates, functions, connective_impls=None, quantifier_impls=None):
    expression = Expression.fromstring(expression)
    visitor = ExpressionVisitor(predicates, functions, connective_impls = connective_impls, quantifier_impls = quantifier_impls )
    return expression.accept(visitor)
