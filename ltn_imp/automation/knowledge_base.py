import torch 
from ltn_imp.fuzzy_operators.aggregators import SatAgg
from ltn_imp.automation.data_loaders import CombinedDataLoader
from ltn_imp.parsing.parser import convert_to_ltn

sat_agg_op = SatAgg()


class KnowledgeBase:
    def __init__(self, expressions, rule_to_data_loader_mapping, predicates={}, functions={}, connective_impls=None, quantifier_impls=None, lr = 0.001):
        self.expressions = expressions
        self.predicates = predicates
        self.functions = functions
        self.connective_impls = connective_impls
        self.quantifier_impls = quantifier_impls
        self.rule_to_data_loader_mapping = rule_to_data_loader_mapping
        self.set_rules()

        try:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        except:
            print("No parameters to optimize")

    def set_rules(self):
        self.declerations = {}
        self.rules = [ convert_to_ltn(rule, predicates=self.predicates,
                                    functions=self.functions, connective_impls=self.connective_impls, 
                                    quantifier_impls=self.quantifier_impls, declerations =  self.declerations) for rule in self.expressions ]
        
        self.rule_to_data_loader_mapping = { self.rules[i] : self.rule_to_data_loader_mapping[expression] for i, expression in enumerate( self.rule_to_data_loader_mapping) }
    
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

        *data, labels = batch 

        for i, var in enumerate( loader.variables ):

             # TODO: If the dataloader is sending more than one instance ( Classifier(x,y,z) ), they need to be unpacked correctly here
            var_data = data[i] 
            indices = (labels == cls).nonzero(as_tuple=True)[0]
            var_mapping[var] = var_data[indices]

        var_mapping[loader.target] = torch.eye(num_classes)[cls].unsqueeze(0) 

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