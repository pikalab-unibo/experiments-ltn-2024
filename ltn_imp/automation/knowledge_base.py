import torch 
from ltn_imp.fuzzy_operators.aggregators import SatAgg
from ltn_imp.automation.data_loaders import CombinedDataLoader
from ltn_imp.parsing.parser import convert_to_ltn

sat_agg_op = SatAgg()


class KnowledgeBase:
    def __init__(self, rules, rule_to_data_loader_mapping, predicates=None, functions=None, connective_impls=None, quantifier_impls=None, lr = 0.001):
        self.rules = [ convert_to_ltn(rule, predicates=predicates, functions=functions, connective_impls=connective_impls, quantifier_impls=quantifier_impls) for rule in rules ]
        self.predicates = predicates
        self.rule_to_data_loader_mapping = { self.rules[i] : rule_to_data_loader_mapping[expression] for i, expression in enumerate( rule_to_data_loader_mapping) }
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

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
    
    def partition_data(self, var_mapping, batch, loader, num_classes, cls):

        for i, var in enumerate( loader.variables ) :
            
            data, labels = batch # TODO: If the dataloader is sending more than one instance ( Classifier(x,y,z) ), they need to be unpacked correctly here. 

            indices = (labels == cls).nonzero(as_tuple=True)[0]
            var_mapping[loader.target] = torch.eye(num_classes)[cls].unsqueeze(0) 
            var_mapping[var] = data[indices]

    def optimize(self, num_epochs=10, log_steps=10):
        all_loaders = set(loader for loaders in self.rule_to_data_loader_mapping.values() for loader in loaders)

        combined_loader = CombinedDataLoader(list(all_loaders))

        for epoch in range(num_epochs):
            for i in range(len(combined_loader)):
                rule_outputs = []
                current_batches = next(combined_loader)

                for rule in self.rules:
                    loaders = self.rule_to_data_loader_mapping[rule]
                    max_classes = max(loader.num_classes for loader in loaders)

                    var_mapping = {}
                    
                    for cls in range(max_classes):
                        for loader in loaders:
                            if cls < loader.num_classes: 
                                batch = current_batches[loader]
                                self.partition_data(var_mapping, batch, loader, loader.num_classes, cls)
                                
                        rule_output = rule(var_mapping)
                        rule_outputs.append(rule_output)

                self.optimizer.zero_grad()
                loss = self.loss(rule_outputs)
                loss.backward()
                self.optimizer.step()

            if epoch % log_steps == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
                print()