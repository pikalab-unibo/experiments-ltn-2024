from nltk.sem import logic
from ltn_imp.fuzzy_operators.connectives import AndConnective, OrConnective, NotConnective, ImpliesConnective, IffConnective, EqConnective
from ltn_imp.fuzzy_operators.predicates import Predicate
from ltn_imp.fuzzy_operators.quantifiers import ForallQuantifier, ExistsQuantifier
from ltn_imp.fuzzy_operators.functions import Function
from ltn_imp.visitor import Visitor, make_visitable
from nltk.sem.logic import Expression
import ltn_imp.fuzzy_operators.connectives as Connectives
from ltn_imp.parsing.expression_transformations import transform
import torch 

make_visitable(logic.Expression)

class LessThan():
    def __init__(self):
        pass

    def __call__(self, tensor1, tensor2):
        return self.forward(tensor1, tensor2)

    def forward(self, tensor1, tensor2):
        return torch.lt(tensor1, tensor2).float()

class MoreThan():
    def __init__(self):
        pass

    def __call__(self, tensor1, tensor2):
        return self.forward(tensor1, tensor2)

    def forward(self, tensor1, tensor2):
        return torch.gt(tensor1, tensor2).float()
    
class Add():
    def __init__(self):
        pass

    def __call__(self, tensor1, tensor2):
        return self.forward(tensor1, tensor2)

    def forward(self, tensor1, tensor2):
        return torch.add(tensor1, tensor2).float()

class Subtract():
    def __init__(self):
        pass

    def __call__(self, tensor1, tensor2):
        return self.forward(tensor1, tensor2)

    def forward(self, tensor1, tensor2):
        return torch.sub(tensor1, tensor2).float()

class Multiply():
    def __init__(self):
        pass

    def __call__(self, tensor1, tensor2):
        return self.forward(tensor1, tensor2)

    def forward(self, tensor1, tensor2):
        return torch.mul(tensor1, tensor2).float()

class Divide():
    def __init__(self):
        pass

    def __call__(self, tensor1, tensor2):
        return self.forward(tensor1, tensor2)

    def forward(self, tensor1, tensor2):
        return torch.div(tensor1, tensor2).float()
    
def get_subclass_with_prefix(module, superclass: type, prefix: str = "default"):
    prefix = prefix.lower()
    for k in dir(module):
        obj = getattr(module, k)
        if isinstance(obj, type) and k.lower().startswith(prefix) and issubclass(obj, superclass):
            return obj()
    
    raise KeyError(f'No subtype of {superclass} found in module {module} with prefix "{prefix}"')


class ExpressionVisitor(Visitor):

    def __init__(self, predicates, functions, connective_impls=None, quantifier_impls=None, declerations = None, declearars = None):
        connective_impls = connective_impls or {}
        quantifier_impls = quantifier_impls or {}

        self.predicates = predicates
        self.functions = functions

        if declerations is None:
            self.declerations = {}
        else:
            self.declerations = declerations

        if declearars is None:
            self.declerars = {}
        else:
            self.declerars = declearars


        And = get_subclass_with_prefix(module=Connectives, superclass=AndConnective, prefix=connective_impls.get('and', 'default'))
        Or = get_subclass_with_prefix(module=Connectives, superclass=OrConnective, prefix=connective_impls.get('or', 'default'))
        Not = get_subclass_with_prefix(module=Connectives, superclass=NotConnective, prefix=connective_impls.get('not', 'default'))
        Implies = get_subclass_with_prefix(module=Connectives, superclass=ImpliesConnective, prefix=connective_impls.get('implies', 'default'))
        Equiv = get_subclass_with_prefix(module=Connectives, superclass=IffConnective, prefix=connective_impls.get('iff', 'default'))
        Eq = get_subclass_with_prefix(module=Connectives, superclass=EqConnective, prefix=connective_impls.get('eq', 'default'))

        Exists = ExistsQuantifier(method=quantifier_impls.get('exists', 'pmean'))
        Forall = ForallQuantifier(method=quantifier_impls.get('forall', 'min'))

        self.connective_map = {
            logic.AndExpression: And,
            logic.OrExpression: Or,
            logic.ImpExpression: Implies,
            logic.IffExpression: Equiv,
            logic.NegatedExpression : Not, 
            logic.EqualityExpression : Eq
        }

        self.quantifier_map = {
            logic.ExistsExpression: Exists,
            logic.AllExpression: Forall
        }

    def handle_predicate(self, variables, functor, var_mapping,  expression):
        results = []

        for var in variables:
            try:
                # Attempt to retrieve and process each variable
                variable_value = self.visit(var)(var_mapping)
                results.append(variable_value)

            except KeyError as e: # If the key is not found, it is a declared variable. 
                if var in self.declerations: # TODO: Either a declared variable is being used ( Which is fine ) or its trying get re-declared ( Which is not fine )
                    if self.declerars[str(var)] != expression:
                        results.append(self.declerations[str(var)])
                    else:
                        continue
                else:
                    self.declerations[str(var)] = Predicate(self.predicates[functor])(*results)
                    self.declerars[str(var)] = expression 
                    return torch.tensor([1.0])
            
        if functor in self.predicates:
            return Predicate(self.predicates[functor])(*results)
        else:
            raise ValueError(f"Unknown functor: {functor}")    

    def handle_function(self, variables, functor, var_mapping, expression):
        results = []
                
        for var in variables:
            try:
                # Attempt to retrieve and process each variable
                variable_value = self.visit(var)(var_mapping)
                results.append(variable_value)

            except KeyError as e: # If the key is not found, it is a declared variable. 
                if var in self.declerations: # TODO: Either a declared variable is being used ( Which is fine ) or its trying get re-declared ( Which is not fine )
                    if self.declerars[str(var)] != expression:
                        results.append(self.declerations[str(var)])
                    else:
                        continue
                else:
                    self.declerations[str(var)] = Function(self.functions[functor])(*results)
                    self.declerars[str(var)] = expression 
                    return torch.tensor([1.0])
            
        if functor in self.functions:
            return Function(self.functions[functor])(*results)
        else:
            raise ValueError(f"Unknown functor: {functor}")      

    def visit_ApplicationExpression(self, expression):

        variables = []
        for arg in expression.args:
            variables.append(arg)

        functor = expression.function
        while hasattr(functor, 'function'):
            functor = functor.function
        functor = str(functor)

        if functor in self.predicates:
            return lambda var_mapping: self.handle_predicate(variables=variables, functor=functor, var_mapping=var_mapping,  expression=expression)
        elif functor in self.functions:
            return lambda var_mapping: self.handle_function(variables=variables, functor=functor, var_mapping=var_mapping, expression=expression)
        else:
            raise ValueError(f"Unknown functor: {functor}")

    def delay_execution(self, left, right, var_mapping, connective, expression):

        try:
            left_value = left(var_mapping)
            right_value = right(var_mapping)
        except:
            right_value = right(var_mapping)
            left_value = left(var_mapping)

        return connective(left_value, right_value)
        
    def visit_BinaryExpression(self, expression):
        connective = self.connective_map.get(type(expression))
        if connective:
            left = self.visit(expression.first)
            right = self.visit(expression.second)
            return lambda var_mapping: self.delay_execution(left, right, var_mapping, connective, expression)
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
        
    def handle_constant(self, variable_mapping, expression):
        try:
            return torch.tensor(float(str(expression)))
        except:
            return torch.tensor(variable_mapping[str(expression)])
        
    def visit_ConstantExpression(self, expression):
        return lambda variable_mapping: self.handle_constant(variable_mapping, expression)

    def handle_variable(self, variable_mapping, expression):
        var = list(expression.variables())[0]
        try:
            return variable_mapping[str(var)]
        except KeyError as e:
            try:
                return self.declerations[str(var)]
            except KeyError as e:
                raise KeyError(f"Variable {var} not recognized")
    
    def visit_IndividualVariableExpression(self, expression):
        return lambda variable_mapping: self.handle_variable(variable_mapping, expression)


def convert_to_ltn(expression, predicates = {}, functions = {}, connective_impls=None, quantifier_impls=None, declerations = None, declerars = None):

    functions["lt"] = LessThan()
    functions["mt"] = MoreThan()
    functions["add"] = Add()
    functions["sub"] = Subtract()
    functions["mul"] = Multiply()
    functions["div"] = Divide()

            
    expression = Expression.fromstring(transform(expression))

    visitor = ExpressionVisitor(predicates, functions, connective_impls = connective_impls, quantifier_impls = quantifier_impls, declerations = declerations, declearars=declerars )
    return expression.accept(visitor)
