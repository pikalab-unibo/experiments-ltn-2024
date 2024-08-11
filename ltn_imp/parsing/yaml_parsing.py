import yaml

def parse_yaml(file):

    data = yaml.safe_load(file)

    instances = data['instances']
    features = data['features']
    constants = data['constants']
    learnable = data['learnable']
    knowledge = data['knowledge']
    constraints = data['constraints']

    print("Instances:", instances)
    print("Features:", features)
    print("Constants:", constants)
    print("Learnable:", learnable)
    print("Knowledge:", knowledge)
    print("Constraints:", constraints)

    return data