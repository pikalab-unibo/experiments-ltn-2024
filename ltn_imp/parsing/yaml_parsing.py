import yaml

def parse_yaml(file):
    data = yaml.safe_load(file)
    return data