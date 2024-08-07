from nltk.sem import logic
import nltk.sem.logic
from ltn_imp.fuzzy_operators.connectives import AndConnective, OrConnective, NotConnective, ImpliesConnective, IffConnective, EqConnective, AddConnective, SubtractConnective, MultiplyConnective, DivideConnective, LessThanConnective, MoreThanConnective, LessThanOrEqualConnective, MoreThanOrEqualConnective, NegativeConnective
from ltn_imp.fuzzy_operators.predicates import Predicate
from ltn_imp.fuzzy_operators.quantifiers import ForallQuantifier, ExistsQuantifier
from ltn_imp.fuzzy_operators.functions import Function
from ltn_imp.visitor import Visitor, make_visitable
import ltn_imp.fuzzy_operators.connectives as Connectives
import torch
import nltk
from torchviz import make_dot
from collections import defaultdict
from nltk.sem.logic import LogicParser
import nltk.sem.logic as nl

make_visitable(logic.Expression)    


ARITH_OPS_LOW_PRESEDENCE = ["+", "-"]
ARITH_OPS_HIGH_PRESEDENCE = ["*", "/"]
EQUALITY_COMP_OPS = ["<=", ">="]
COMP_OPS = ["<", ">"]

nl.Tokens.OR_LIST += ARITH_OPS_LOW_PRESEDENCE 
nl.Tokens.AND_LIST += ARITH_OPS_HIGH_PRESEDENCE
nl.Tokens.EQ_LIST += EQUALITY_COMP_OPS + COMP_OPS

class ArithExpression(nl.BinaryExpression):
    def __init__(self, op, first, second):
        super().__init__(first, second)
        self.op = op

    def getOp(self):
        return self.op
    
    @classmethod
    def factory_for(cls, op):
        def factory(first, second):
            return cls(first, second)
        return factory 
    
class AdditionExpression(ArithExpression):
    def __init__(self, first, second):
        super().__init__('+', first, second)

class SubtractionExpression(ArithExpression):
    def __init__(self, first, second):
        super().__init__('-', first, second)

class MultiplicationExpression(ArithExpression):
    def __init__(self, first, second):
        super().__init__('*', first, second)

class DivisionExpression(ArithExpression):
    def __init__(self, first, second):
        super().__init__('/', first, second)

class MoreThanExpression(ArithExpression):
    def __init__(self, first, second):
        super().__init__('>', first, second)

class LessThanExpression(ArithExpression):
    def __init__(self, first, second):
        super().__init__('<', first, second)

class LessEqualExpression(ArithExpression):
    def __init__(self, first, second):
        super().__init__('<=', first, second)

class MoreEqualExpression(ArithExpression):
    def __init__(self, first, second):
        super().__init__('>=', first, second)

class NegativeExpression(nl.NegatedExpression):
    def __init__(self, term):
        assert isinstance(term, nl.Expression), f"{term} is not an Expression"
        self.term = term

    @property
    def type(self):
        return self.term.type

    def _set_type(self, other_type=nl.ANY_TYPE, signature=None):
        self.term._set_type(other_type, signature)

    def findtype(self, variable):
        return self.term.findtype(variable)

    def visit(self, function, combinator):
        return combinator([function(self.term)])

    def __eq__(self, other):
        return isinstance(other, NegativeExpression) and self.term == other.term

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.term)

    def __str__(self):
        return f"-{self.term}"


class LessEqualExpression(nl.BinaryExpression):
    def getOp(self):
        return "<="

class MoreEqualExpression(nl.BinaryExpression):
    def getOp(self):
        return ">="
    

class LogicParser(nl.LogicParser):

    def __init__(self, type_check=False):
        super().__init__(type_check)

        self.operator_precedence = dict(
            [(x, 1) for x in nl.Tokens.LAMBDA_LIST] +                # Lambda (λ)
            [(nl.APP, 2)] +                                             # Application (APP)
            [(x, 3) for x in nl.Tokens.NOT_LIST] +                   # Negation (-)
            [(x, 5) for x in nl.Tokens.AND_LIST] +                   # Conjunction (and, &)
            [(x, 6) for x in nl.Tokens.OR_LIST] +                   # Disjunction (or, |)
            [(x, 7) for x in nl.Tokens.IMP_LIST] +                  # Implication (implies, ->)
            [(x, 8) for x in nl.Tokens.IFF_LIST] +                  # Biconditional (iff, <->)
            [(x, 9) for x in nl.Tokens.EQ_LIST + nl.Tokens.NEQ_LIST] +  # Equality (=) and Inequality (!=)
            [(x, 10) for x in nl.Tokens.QUANTS] +                     # Quantifiers (some, all, iota)
            [(None, 11)]                                             # Default (None)
        )
        self.right_associated_operations = [nl.APP]

    def get_BooleanExpression_factory(self, tok):
        if tok == '+':
            return AdditionExpression.factory_for('+')
        elif tok == '-':
            return SubtractionExpression.factory_for('-')
        elif tok == '*':
            return MultiplicationExpression.factory_for('*')
        elif tok == '/':
            return DivisionExpression.factory_for('/')
        elif tok == '>':
            return MoreThanExpression.factory_for('>')
        elif tok == '<':
            return LessThanExpression.factory_for('<')
                
        return super().get_BooleanExpression_factory(tok)
    
    def handle_negation(self, tok, context):
        if tok == '-':
            if self.inRange(0):
                if self.token(0).isdigit():
                    term = self.token()
                    return NegativeExpression(nl.ConstantExpression(nl.Variable(term)))
                elif self.token(0) == nl.Tokens.OPEN:
                    self.token()  # Consume the '('
                    term = self.process_next_expression(None)
                    self.assertNextToken(nl.Tokens.CLOSE)  # Ensure there's a closing ')'
                    return NegativeExpression(term)
                else:
                    term = self.process_next_expression(tok)
                    return NegativeExpression(term)
        return super().handle_negation(tok, context)
    

    def attempt_EqualityExpression(self, expression, context):
        if self.inRange(0):
            tok = self.token(0)
            
            if tok in nl.Tokens.EQ_LIST + nl.Tokens.NEQ_LIST  and self.has_priority(tok, context):
                if tok == "=":
                    self.token()
                    expression = self.make_EqualityExpression(expression, self.process_next_expression(tok))

                elif tok == "!=":
                    self.token()
                    expression = self.make_EqualityExpression(expression, self.process_next_expression(tok))
                    expression = self.make_NegatedExpression(expression)

                elif tok == "<":
                    tok = self.token(1)

                    if tok == "=":
                        self.token()
                        self.token()
                        return LessEqualExpression(expression, self.process_next_expression(tok))
                    
                    else:
                        if tok == nl.Tokens.OPEN:
                            self.token()
                            return LessThanExpression(expression, self.process_next_expression(None))
                        
                        elif tok == "-":
                            self.token()
                            return LessThanExpression(expression, self.process_next_expression(None))
                        
                        else:
                            tok = self.token(0)
                            self.process_next_expression(tok)
                            return LessThanExpression(expression, self.process_next_expression(tok))
                    
                elif tok == ">":
                    tok = self.token(1)

                    if tok == "=":
                        self.token()
                        self.token()
                        return MoreEqualExpression(expression, self.process_next_expression(tok))
                    
                    else:
                        if tok == nl.Tokens.OPEN:
                            self.token()
                            return MoreThanExpression(expression, self.process_next_expression(None))
                        
                        elif tok == "-":
                            self.token()
                            return MoreThanExpression(expression, self.process_next_expression(None))
                        
                        else:
                            tok = self.token(0)
                            self.process_next_expression(tok)
                            return MoreThanExpression(expression, self.process_next_expression(tok))
                    
        return expression
    
    def parse(self, data, signature=None):
        return super().parse(data, signature)

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
            print(f"For Expression {self.expression} this error occured: {e}")
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

        Add = get_subclass_with_prefix(module=Connectives, superclass=AddConnective, prefix=functions.get('add', 'default'))
        Subtract = get_subclass_with_prefix(module=Connectives, superclass=SubtractConnective, prefix=functions.get('sub', 'default'))
        Multiply = get_subclass_with_prefix(module=Connectives, superclass=MultiplyConnective, prefix=functions.get('mul', 'default'))
        Divide = get_subclass_with_prefix(module=Connectives, superclass=DivideConnective, prefix=functions.get('div', 'default'))

        LessThan = get_subclass_with_prefix(module=Connectives, superclass=LessThanConnective, prefix=predicates.get('lt', 'default'))
        MoreThan = get_subclass_with_prefix(module=Connectives, superclass=MoreThanConnective, prefix=predicates.get('gt', 'default'))
        LessThanEqual = get_subclass_with_prefix(module=Connectives, superclass=LessThanOrEqualConnective, prefix=predicates.get('le', 'default'))
        MoreThanEqual = get_subclass_with_prefix(module=Connectives, superclass=MoreThanOrEqualConnective, prefix=predicates.get('ge', 'default'))

        Negative = get_subclass_with_prefix(module=Connectives, superclass=NegativeConnective, prefix=connective_impls.get('neg', 'default'))
        
        self.connective_map = {
            logic.AndExpression: And,
            logic.OrExpression: Or,
            logic.ImpExpression: Implies,
            logic.IffExpression: Equiv,
            logic.NegatedExpression: Not,
            NegativeExpression: Negative,
            logic.EqualityExpression: Eq,
            AdditionExpression: Add,
            SubtractionExpression: Subtract,
            MultiplicationExpression: Multiply,
            DivisionExpression: Divide,
            LessThanExpression: LessThan,
            MoreThanExpression: MoreThan,

            LessEqualExpression: LessThanEqual, # Still Problematic
            MoreEqualExpression: MoreThanEqual  # Still Problematic
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
                
            self.intermediate_results[functor].append(torch.tensor([1.0], requires_grad=True))
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

    def parse(self, expression):
        parser = LogicParser()
        expression = parser.parse(expression)
        self.expression = expression
        return expression
        
    def __call__(self, expression):

        parser = LogicParser()
        expression = parser.parse(expression)
    
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
