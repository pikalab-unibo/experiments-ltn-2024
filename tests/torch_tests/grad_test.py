import torch
import unittest
from ltn_imp.parsing.parser import LTNConverter
from ltn_imp.fuzzy_operators.predicates import Predicate

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

class GradTests(unittest.TestCase):

    def setUp(self):
        self.model = SimpleModel()
        self.predicates = {'Classifier': Predicate(self.model)}
        self.connective_impls = {
            'imp': 'luk',
            'eq': 'default'
        }
        self.quantifier_impls = {
            'forall': 'min'
        }
        self.converter = LTNConverter(
            predicates=self.predicates,
            connective_impls=self.connective_impls,
            quantifier_impls=self.quantifier_impls
        )

    def test_less_than_preserves_graph(self):
        expression = 'all x. (Classifier(x) < 1)'
        data = torch.tensor([[0.2, 0.8], [0.5, 0.5]], requires_grad=True)
        var_mapping = {'x': data}
        rule = self.converter(expression)
        result = rule(var_mapping)
        loss = 1 - result.mean()
        loss.backward()
        self.assertIsNotNone(data.grad)
        print("Grad for less_than_preserves_graph:", data.grad)

    def test_more_than_preserves_graph(self):
        expression = 'all x. (Classifier(x) > 0)'
        data = torch.tensor([[0.2, 0.8], [0.5, 0.5]], requires_grad=True)
        var_mapping = {'x': data}
        rule = self.converter(expression)
        result = rule(var_mapping)
        loss = 1 - result.mean()
        loss.backward()
        self.assertIsNotNone(data.grad)
        print("Grad for more_than_preserves_graph:", data.grad)

    def test_combined_expression_preserves_graph(self):
        expression = 'all x. Classifier(x) > 0 and Classifier(x) < 1'
        data = torch.tensor([[0.2, 0.8], [0.5, 0.5]], requires_grad=True)
        var_mapping = {'x': data}
        rule = self.converter(expression)
        result = rule(var_mapping)
        loss = 1 - result.mean()
        loss.backward()
        self.assertIsNotNone(data.grad)
        print("Grad for combined_expression_preserves_graph:", data.grad)

    def test_implies_preserves_graph(self):
        expression = 'all x. (Classifier(x) -> Classifier(x) > 0)'
        data = torch.tensor([[0.2, 0.8], [0.5, 0.5]], requires_grad=True)
        var_mapping = {'x': data}
        rule = self.converter(expression)
        result = rule(var_mapping)
        loss = 1 - result.mean()
        loss.backward()
        self.assertIsNotNone(data.grad)
        print("Grad for implies_preserves_graph:", data.grad)

    def test_variable_declaration_and_operations(self):
        expression = 'all x. (Classifier(x,z) and z > 0)'
        data = torch.tensor([[0.2, 0.8], [0.5, 0.5]], requires_grad=True)
        var_mapping = {'x': data}
        rule = self.converter(expression)
        result = rule(var_mapping)
        z = rule.visitor.declarations.get('z', None)
        loss = 1 - result.mean()
        loss.backward()
        self.assertIsNotNone(data.grad)
        self.assertIsNotNone(z)
        print("Grad for variable_declaration_and_operations:", data.grad)
        print("Value of z:", z)

if __name__ == "__main__":
    unittest.main()
