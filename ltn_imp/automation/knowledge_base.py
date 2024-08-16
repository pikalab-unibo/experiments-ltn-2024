import torch 
from ltn_imp.fuzzy_operators.aggregators import SatAgg
from ltn_imp.automation.data_loaders import CombinedDataLoader, LoaderWrapper
from ltn_imp.parsing.parser import LTNConverter
from ltn_imp.parsing.ancillary_modules import ModuleFactory
from ltn_imp.automation.network_factory import NNFactory
import yaml


sat_agg_op = SatAgg()

class KnowledgeBase:
    def __init__(self, yaml_file, loaders=None, constant_mapping=None):
        with open(yaml_file, "r") as file:
            config = yaml.safe_load(file)
        self.config = config
        self.factory = NNFactory()

        self.set_predicates()
        self.converter = LTNConverter(yaml=self.config, predicates=self.predicates)
        self.set_rules()
        self.set_ancillary_rules()
        self.set_loaders()
        self.loaders = loaders # TODO: Placeholder
        self.set_rule_to_data_loader_mapping()
        self.constant_mapping = constant_mapping  # TODO: Placeholder


    def evaluate_layer_size(self, layer_size, features_dict, instance_name):
        in_size_str, out_size_str = layer_size
        feature_count = len(features_dict[instance_name])
        in_size = eval(in_size_str.replace('x', str(feature_count)))
        out_size = eval(out_size_str.replace('x', str(feature_count)))
        return in_size, out_size

    def set_predicates(self):
        features = self.config["features"]
        self.predicates = {}

        for predicate_name, predicate_info in self.config["predicates"].items():
            args = predicate_info["args"]
            structure = predicate_info["structure"]

            instance_name = args[0]['in']
            layers = []
            activations = []

            for layer in structure['layers']:
                layer_type = list(layer.keys())[0]  # E.g., 'in', 'hidden', 'out'
                layer_size = layer[layer_type]
                activation = layer.get('activation', None)
                in_size, out_size = self.evaluate_layer_size(layer_size, features, instance_name)
                layers.append((in_size,out_size))
                activations.append(activation)
                
            # Create the network using the NNFactory
            network = self.factory(
                layers = layers,
                activations=activations
            )

            # Store the network in the predicates dictionary
            self.predicates[predicate_name] = network

    def set_rules(self):
        self.rules = [ self.converter(rule) for rule in self.config["constraints"]]

    def set_ancillary_rules(self):
        if "knowledge" not in self.config:
            return
        for anchor in self.config["knowledge"]:
            name = anchor["rule"]
            params = anchor["args"]
            functionality = anchor["clause"]
            ModuleFactory(self.converter).create_module(name, params, functionality)

    def set_loaders(self):
        loaders = []
        for pred in self.predicates.keys():
            args = self.config["predicates"][pred]["args"]
            values = [list(arg.values())[0] for arg in args]
            keys = [list(arg.keys())[0] for arg in args]

            variables = []
            targets = []

            for k, v in zip(keys, values):
                if k == "in":
                    variables.append(v)
                else:
                    targets.append(v)

            loaders.append(LoaderWrapper(loader=..., variables=variables, targets=targets)) # TODO: Not implemented yet

        self.loaders = loaders

    def set_rule_to_data_loader_mapping(self):
        rule_to_loader_mapping = {}

        for rule in self.rules:
            variables = rule.variables()
            for v in variables:
                for loader in self.loaders:
                    if str(v) in loader.variables or str(v) in loader.targets:
                        if rule in rule_to_loader_mapping:
                            rule_to_loader_mapping[rule].append(loader)
                        else:
                            rule_to_loader_mapping[rule] = [loader]

        self.rule_to_data_loader_mapping = rule_to_loader_mapping
    
    def loss(self, rule_outputs):
        # Compute satisfaction level
        sat_agg_value = sat_agg_op(
            *rule_outputs
        )
        # Compute loss
        loss = 1.0 - sat_agg_value
        return loss
    
    def parameters(self):
        params = []
        for model in self.predicates.values():
            if hasattr(model, 'parameters'):
                params += list(model.parameters())
        
        return params
    
    def partition_data(self, var_mapping, batch, loader):

        # Take care of constants TODO: I dont think this needs to get repeated for every batch but for now its fine 
        for k,v in self.constant_mapping.items(): 
            var_mapping[k] = v

        *batch, = batch 

        # Add Variables ( Input Data )
        for i, var in enumerate(loader.variables):
            var_mapping[var] = batch[i]

        # Add Targets ( Output Data )
        for i, target in enumerate(loader.targets):
            var_mapping[target] = batch[i + len(loader.variables)]

    def optimize(self, num_epochs=10, log_steps=10, lr=0.001):

        try:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        except:
            print("No parameters to optimize")

        all_loaders = set(loader for loaders in self.rule_to_data_loader_mapping.values() if loaders is not None for loader in loaders if loader is not None)

        combined_loader = CombinedDataLoader([loader for loader in all_loaders if loader is not None])

        for epoch in range(num_epochs):
            for _ in range(len(combined_loader)):
                rule_outputs = []
                current_batches = next(combined_loader)

                for rule in self.rules:
                    loaders = self.rule_to_data_loader_mapping[rule]
                    var_mapping = {}
                
                    if loaders == None:
                        rule_outputs.append(rule(var_mapping))
                        continue
                        
                    for loader in loaders:
                        batch = current_batches[loader]
                        self.partition_data(var_mapping, batch, loader)
                    
                    rule_output = rule(var_mapping)
                    rule_outputs.append(rule_output)
                            
                self.optimizer.zero_grad()
                loss = self.loss(rule_outputs)
                loss.backward()
                self.optimizer.step()

            if epoch % log_steps == 0:
                print([str(rule) for rule in self.rules])
                print("Rule Outputs: ", rule_outputs)
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
                print()