import torch 
from ltn_imp.fuzzy_operators.aggregators import SatAgg
from sklearn.preprocessing import OneHotEncoder
from ltn_imp.automation.data_loaders import CombinedDataLoader
from ltn_imp.parsing.parser import convert_to_ltn
import numpy as np 

sat_agg_op = SatAgg()

# TODO: How to partition the data for each rule (i.e, how to map the data to the variables in the rules especially when the rules use more than one predicate)

class KnowledgeBase:
    def __init__(self, rules, rule_to_data_loader_mapping, loader_to_variable, loader_classes, loader_to_target, predicates=None, functions=None, connective_impls=None, quantifier_impls=None, lr = 0.001):
        self.rules = [ convert_to_ltn(rule, predicates=predicates, functions=functions, connective_impls=connective_impls, quantifier_impls=quantifier_impls) for rule in rules ]
        self.predicates = predicates
        self.rule_to_data_loader_mapping = { self.rules[i] : rule_to_data_loader_mapping[expression] for i, expression in enumerate( rule_to_data_loader_mapping) }
        self.loader_to_variable = loader_to_variable
        self.loader_to_target = loader_to_target
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loader_classes = loader_classes

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
        
        # Print parameter information
        for param in params:
            print(f"Shape: {param.shape}")
            print(f"Requires Grad: {param.requires_grad}")
            print("-" * 50)
        return params
    
    def partition_data(self, var_mapping, batch, loader, num_classes, cls):

        for i, var in enumerate( self.loader_to_variable[loader] ) :
            data, labels = batch
            indices = (labels == cls).nonzero(as_tuple=True)[0]
            var_mapping[self.loader_to_target[loader]] = torch.eye(num_classes)[cls].unsqueeze(0) 
            var_mapping[var] = data[indices]

    def optimize(self, num_epochs=10, log_steps=10):
        all_loaders = set(loader for loaders in self.rule_to_data_loader_mapping.values() for loader in loaders)

        # Initialize a single CombinedDataLoader for all loaders
        combined_loader = CombinedDataLoader(list(all_loaders))

        for epoch in range(num_epochs):
            for i in range(len(combined_loader)):
                rule_outputs = []
                current_batches = next(combined_loader)

                for rule in self.rules:
                    loaders = self.rule_to_data_loader_mapping[rule]
                    max_classes = max(self.loader_classes[loader] for loader in loaders)

                    var_mapping = {}
                    
                    for cls in range(max_classes):
                        for loader in loaders:
                            if cls < self.loader_classes[loader]: 
                                batch = current_batches[loader]
                                self.partition_data(var_mapping, batch, loader, self.loader_classes[loader], cls)
                                
                        rule_output = rule(var_mapping)
                        rule_outputs.append(rule_output)

                self.optimizer.zero_grad()
                loss = self.loss(rule_outputs)
                loss.backward()
                self.optimizer.step()

            if epoch % log_steps == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
                print()


