import re

def transform_more(expression_str):
    parts = expression_str.split(">")
    if len(parts) == 2:
        first, second = parts[0].strip(), parts[1].strip()
        transformed_expression = f"mt({first},{second})"
        return transformed_expression
    else:
        return expression_str

def transform_more_equal(expression_str):
    parts = expression_str.split(">=")
    if len(parts) == 2:
        first, second = parts[0].strip(), parts[1].strip()
        transformed_expression = f"mt({first},{second}) or ({first} = {second})"
        return transformed_expression
    else:
        return expression_str

def transform_less(expression_str):
    parts = expression_str.split("<")
    if len(parts) == 2:
        first, second = parts[0].strip(), parts[1].strip()
        transformed_expression = f"lt({first},{second})"
        return transformed_expression
    else:
        return expression_str

def transform_less_equal(expression_str):
    parts = expression_str.split("<=")
    if len(parts) == 2:
        first, second = parts[0].strip(), parts[1].strip()
        transformed_expression = f"lt({first},{second}) or ({first} = {second})"
        return transformed_expression
    else:
        return expression_str
    
def transform_add(expression_str):
    parts = expression_str.split("+")
    if len(parts) == 2:
        first, second = parts[0].strip(), parts[1].strip()
        transformed_expression = f"add({first},{second})"
        return transformed_expression
    else:
        return expression_str

def transform_subtract(expression_str):
    parts = expression_str.split("-")
    if len(parts) == 2:
        first, second = parts[0].strip(), parts[1].strip()
        transformed_expression = f"sub({first},{second})"
        return transformed_expression
    else:
        return expression_str

def transform_multiply(expression_str):
    parts = expression_str.split("*")
    if len(parts) == 2:
        first, second = parts[0].strip(), parts[1].strip()
        transformed_expression = f"mul({first},{second})"
        return transformed_expression
    else:
        return expression_str

def transform_divide(expression_str):
    parts = expression_str.split("/")
    if len(parts) == 2:
        first, second = parts[0].strip(), parts[1].strip()
        transformed_expression = f"div({first},{second})"
        return transformed_expression
    else:
        return expression_str
    
def transform_negate(subexpr):
    return f"mul(sub(0,1),{subexpr})"


def transform_encapsulated(expression_str):
    # Updated pattern to ensure the left side of the parentheses is not preceded by a dot
    encapsulated_pattern = re.compile(r'(?<!\.)\(([^()]*?(?:\([^()]*\)[^()]*?)*?[\+\-\*\/<>=][^()]*?(?:\([^()]*\)[^()]*?)*?)\)')
    
    while encapsulated_pattern.search(expression_str):
        expression_str = encapsulated_pattern.sub(lambda x: transform(x.group(1)), expression_str)
        
    return expression_str

def transform_expression(expression_str):
    expression_str = transform_encapsulated(expression_str)
    negation_pattern = re.compile(r'(?<!<)-(\S+)(?<!>)')
    while negation_pattern.search(expression_str):
        expression_str = negation_pattern.sub(lambda x: transform_negate(x.group(1)), expression_str, 1)
        

    patterns = {
        "/": re.compile(r"(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+)\s*/\s*(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+(\.\d+)?|\d+(\.\d+)?)"),
        "*": re.compile(r"(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+)\s*\*\s*(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+(\.\d+)?|\d+(\.\d+)?)"),
        "-": re.compile(r"(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+)\s*-\s*(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+(\.\d+)?|\d+(\.\d+)?)"),
        "+": re.compile(r"(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+)\s*\+\s*(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+(\.\d+)?|\d+(\.\d+)?)"),
        "<=": re.compile(r"(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+)\s*<=\s*(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+(\.\d+)?|\d+(\.\d+)?)"),
        ">=": re.compile(r"(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+)\s*>=\s*(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+(\.\d+)?|\d+(\.\d+)?)"),
        "<": re.compile(r"(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+)\s*<\s*(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+(\.\d+)?|\d+(\.\d+)?)"),
        ">": re.compile(r"(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+)\s*>\s*(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+(\.\d+)?|\d+(\.\d+)?)"),
    }
    
    for op, pattern in patterns.items():
        matches = pattern.findall(expression_str)
        for match in matches:
            subexpr = f"{match[0]} {op} {match[1]}"
            if op == "+":
                transformed_subexpr = transform_add(subexpr)
            elif op == "-":
                transformed_subexpr = transform_subtract(subexpr)
            elif op == "*":
                transformed_subexpr = transform_multiply(subexpr)
            elif op == "/":
                transformed_subexpr = transform_divide(subexpr)
            elif op == "<=":
                transformed_subexpr = transform_less_equal(subexpr)
            elif op == ">=":
                transformed_subexpr = transform_more_equal(subexpr)
            elif op == "<":
                transformed_subexpr = transform_less(subexpr)
            elif op == ">":
                transformed_subexpr = transform_more(subexpr)
            expression_str = expression_str.replace(subexpr, transformed_subexpr)

    return expression_str


def transform(expression):
    operators = {"+", "-", "*", "/", "<", "<=", ">", ">="}
    
    if expression.startswith("all") or expression.startswith("exist"):
        parts = expression.split('.', 1)
        quantifier = parts[0]
        rest = parts[1]
    else:
        quantifier = ""
        rest = expression

    before = rest
    after = transform_expression(rest)
    
    while before != after:
        if any(op in after for op in operators):
            before, after = after, transform_expression(after)
        else:
            break

    if quantifier:
        return f"{quantifier}. ({after})"
    else:
        return after