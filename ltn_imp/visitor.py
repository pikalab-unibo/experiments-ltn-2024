class Visitor:
    def visit(self, node):
        for cls in type(node).mro():
            method_name = 'visit_' + cls.__name__
            if hasattr(self, method_name):
                visitor = getattr(self, method_name)
                return visitor(node)
        raise ValueError(f"No visitor for {node} of type {type(node)} found in {self}.")



def accept(self, visitor: Visitor):
    assert isinstance(visitor, Visitor), f"Expected a {Visitor.__name__} object, got {type(visitor)}"
    return visitor.visit(self)


def make_visitable(cls: type, accept_method_name: str = 'accept'):
    setattr(cls, accept_method_name, accept)
