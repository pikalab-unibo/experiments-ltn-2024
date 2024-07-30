from nltk.sem import logic
import nltk.sem.logic
from ltn_imp.fuzzy_operators.connectives import AndConnective, OrConnective, NotConnective, ImpliesConnective, IffConnective, EqConnective
from ltn_imp.fuzzy_operators.predicates import Predicate
from ltn_imp.fuzzy_operators.quantifiers import ForallQuantifier, ExistsQuantifier
from ltn_imp.fuzzy_operators.functions import Function
from ltn_imp.visitor import Visitor, make_visitable
from nltk.sem.logic import Expression
import ltn_imp.fuzzy_operators.connectives as Connectives
from ltn_imp.parsing.expression_transformations import transform
import torch
import nltk
from torchviz import make_dot
from collections import defaultdict

make_visitable(logic.Expression)

class LessThan:
    def __init__(self, k=10):
        self.k = k

    def forward(self, tensor1, tensor2):
        return torch.sigmoid(self.k * (tensor2 - tensor1))

    def __call__(self, tensor1, tensor2):
        return self.forward(tensor1, tensor2)
class MoreThan:
    def __init__(self, k=10):
        self.k = k

    def forward(self, tensor1, tensor2):
        return torch.sigmoid(self.k * (tensor1 - tensor2))
    
    def __call__(self, tensor1, tensor2):
        return self.forward(tensor1, tensor2)
    
class Add:
    def forward(self, tensor1, tensor2):
        return torch.add(tensor1, tensor2).float()
    
    def __call__(self, tensor1, tensor2):
        return self.forward(tensor1, tensor2)
    
class Subtract:
    def forward(self, tensor1, tensor2):
        return torch.sub(tensor1, tensor2).float()
    
    def __call__(self, tensor1, tensor2):
        return self.forward(tensor1, tensor2)
class Multiply:
    def forward(self, tensor1, tensor2):
        return torch.mul(tensor1, tensor2).float()
    
    def __call__(self, tensor1, tensor2):
        return self.forward(tensor1, tensor2)
class Divide:
    def forward(self, tensor1, tensor2):
        return torch.div(tensor1, tensor2).float()
    
    def __call__(self, tensor1, tensor2):
        return self.forward(tensor1, tensor2)
    
def get_subclass_with_prefix(module, superclass: type, prefix: str = "default"):
    prefix = prefix.lower()
    for k in dir(module):
        obj = getattr(module, k)
        if isinstance(obj, type) and k.lower().startswith(prefix) and issubclass(obj, superclass):
            return obj()
    
    raise KeyError(f'No subtype of {superclass} found in module {module} with prefix "{prefix}"')

class ConvertedExpression:
    def __init__(self, expression, converted, visitor):
        self.expression = expression
        self.converted = converted
        self.visitor = visitor

    def __call__(self, *args, **kwargs):
        try:
            return self.converted(*args, **kwargs)
        except Exception as e:
            print(e)
            print(self.expression)
            raise e
        
    def __str__(self):
        return str(self.expression)

    def comp_graph(self, var_mapping):
        # Get the final result
        result = self(var_mapping)
        # Collect all intermediate results with requires_grad info
        params = {
            f'{name}_{i} (requires_grad={param.requires_grad})': param 
            for name, param_list in self.visitor.intermediate_results.items() 
            for i, param in enumerate(param_list)
        }
        # Visualize the computation graph
        dot = make_dot(result, params=params)
        return dot
    
class ExpressionVisitor(Visitor):
    def __init__(self, predicates, functions, connective_impls=None, quantifier_impls=None, declarations=None, declarers=None):
        connective_impls = connective_impls or {}
        quantifier_impls = quantifier_impls or {}

        self.predicates = predicates
        self.functions = functions

        self.declarations = declarations if declarations is not None else {}
        self.declarers = declarers if declarers is not None else {}

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
            logic.NegatedExpression: Not,
            logic.EqualityExpression: Eq
        }

        self.quantifier_map = {
            logic.ExistsExpression: Exists,
            logic.AllExpression: Forall
        }

        self.intermediate_results = defaultdict(list)

    def handle_predicate(self, variables, functor, var_mapping, expression):
        inputs = []
        predicate = Predicate(self.predicates[functor])

        to_be_declared = None

        for i, var in enumerate(variables):

            if type(var) != nltk.sem.logic.IndividualVariableExpression and type(var) != nltk.sem.logic.ConstantExpression:
                var = self.visit(var)(var_mapping)
                inputs.append(var)
                continue
    
            var = str(var)

            if var in var_mapping:
                inputs.append(var_mapping[var])
            elif var in self.declarations:
                if self.declarers[var] == str(expression):
                    to_be_declared = variables[i:]
                    break
                else:
                    value = self.declarations[var]
                    inputs.append(value)
            else:
                to_be_declared = variables[i:]
                break

        results = predicate(*inputs)
        
        if not isinstance(results, tuple) and results.dim() == 0:
            results = results.unsqueeze(0)
            
        if to_be_declared is not None:
            for i, var in enumerate(to_be_declared):
                value = results[i]
                self.declarations[str(var)] = value
                self.declarers[str(var)] = str(expression)
            return torch.tensor([1.0], requires_grad=True)

        # Ensuring results is a tensor
        if isinstance(results, (list, tuple)):
            results_tensor = torch.stack([res if isinstance(res, torch.Tensor) else torch.tensor(res) for res in results])
        else:
            results_tensor = results

        self.intermediate_results[functor].append(results_tensor)
        return results_tensor


    def handle_function(self, variables, functor, var_mapping, expression):
        inputs = []
        func = Function(self.functions[functor])

        to_be_declared = None

        for i, var in enumerate(variables):

            if type(var) != nltk.sem.logic.IndividualVariableExpression:
                var = self.visit(var)(var_mapping)
                inputs.append(var)
                continue
    
            var = str(var)

            if var in var_mapping:
                inputs.append(var_mapping[var])
            elif var in self.declarations:
                if self.declarers[var] == str(expression):
                    to_be_declared = variables[i:]
                    break
                else:
                    value = self.declarations[var]
                    inputs.append(value)
            else:
                to_be_declared = variables[i:]
                break

        results = func(*inputs)
        
        if not isinstance(results, tuple) and results.dim() == 0:
            results = results.unsqueeze(0)
            
        if to_be_declared is not None:
            for i, var in enumerate(to_be_declared):
                value = results[i]
                self.declarations[str(var)] = value
                self.declarers[str(var)] = str(expression)
            return torch.tensor([1.0], requires_grad=True)

        # Ensuring results is a tensor
        if isinstance(results, (list, tuple)):
            results_tensor = torch.stack([res if isinstance(res, torch.Tensor) else torch.tensor(res) for res in results])
        else:
            results_tensor = results

        self.intermediate_results[functor].append(results_tensor)
        return results_tensor



    def visit_ApplicationExpression(self, expression):
        variables = [arg for arg in expression.args]
        functor = expression
        
        while hasattr(functor, 'function'):
            functor = functor.function

        functor = str(functor)
        
        if functor in self.predicates:
            return ConvertedExpression(expression, lambda var_mapping: self.handle_predicate(variables, functor, var_mapping, expression), self)
        elif functor in self.functions:
            return ConvertedExpression(expression, lambda var_mapping: self.handle_function(variables, functor, var_mapping, expression), self)
        else:
            raise ValueError(f"Unknown functor: {functor}")

    def delay_execution(self, left, right, var_mapping, connective, expression):
        try: # Right side might be declaring a variable in the left 
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
            return ConvertedExpression(expression, lambda var_mapping: self.delay_execution(left, right, var_mapping, connective, expression), self)
        else:
            raise NotImplementedError(f"Unsupported binary expression type: {type(expression)}")

    def visit_NegatedExpression(self, expression):
        connective = self.connective_map.get(type(expression))
        term = self.visit(expression.term)
        return ConvertedExpression(expression, lambda var_mapping: connective(term.converted(var_mapping)), self)

    def visit_QuantifiedExpression(self, expression):
        quantifier = self.quantifier_map.get(type(expression))
        if quantifier:
            term = self.visit(expression.term)
            return ConvertedExpression(expression, lambda variable_mapping: quantifier(term.converted(variable_mapping)), self)
        else:
            raise NotImplementedError(f"Unsupported quantifier expression type: {type(expression)}")
                
    def handle_variable(self, variable_mapping, expression):
        var = list(expression.variables())[0]
        if str(var) in variable_mapping:
            return variable_mapping[str(var)]
        elif str(var) in self.declarations:
            return self.declarations[str(var)]
        else:
            raise KeyError(f"Variable {var} not recognized")
    
    def visit_IndividualVariableExpression(self, expression):
        return ConvertedExpression(expression, lambda variable_mapping: self.handle_variable(variable_mapping, expression), self)

    def handle_constant(self, variable_mapping, expression):
        if str(expression) in variable_mapping:
            return variable_mapping[str(expression)]
        else:
            return torch.tensor(float(str(expression)),requires_grad=True)
                
    def visit_ConstantExpression(self, expression):
        return ConvertedExpression(expression, lambda variable_mapping: self.handle_constant(variable_mapping, expression), self)

class LTNConverter:
    def __init__(self, predicates={}, functions={}, connective_impls=None, quantifier_impls=None, declarations={}, declarers={}):
        self.predicates = predicates
        self.functions = functions
        self.connective_impls = connective_impls
        self.quantifier_impls = quantifier_impls
        self.declarations = declarations
        self.declarers = declarers
        self.expression = None

        # Adding default functions
        self.functions["lt"] = LessThan()
        self.functions["mt"] = MoreThan()
        self.functions["add"] = Add()
        self.functions["sub"] = Subtract()
        self.functions["mul"] = Multiply()
        self.functions["div"] = Divide()

    def __call__(self, expression, process=True):
        if process:
            expression = Expression.fromstring(transform(expression))
        else: 
            expression = Expression.fromstring(expression)
                                               
        self.expression = expression
        visitor = ExpressionVisitor(
            self.predicates, 
            self.functions, 
            connective_impls=self.connective_impls, 
            quantifier_impls=self.quantifier_impls, 
            declarations=self.declarations, 
            declarers=self.declarers
        )
        return ConvertedExpression(self.expression, expression.accept(visitor), visitor)
