import torch 
from ltn_imp.fuzzy_operators.aggregators import SatAgg
from ltn_imp.automation.data_loaders import CombinedDataLoader
from ltn_imp.parsing.parser import LTNConverter
from ltn_imp.parsing.ancillary_modules import ModuleFactory

sat_agg_op = SatAgg()


class KnowledgeBase:
    def __init__(self, learning_rules, ancillary_rules, rule_to_data_loader_mapping, predicates={}, functions={}, connective_impls=None, quantifier_impls=None, lr = 0.001, constant_mapping = {}):

        self.learning_rules = learning_rules
        self.ancillary_rules = ancillary_rules
        self.predicates = predicates
        self.functions = functions
        self.connective_impls = connective_impls
        self.quantifier_impls = quantifier_impls
        self.rule_to_data_loader_mapping = rule_to_data_loader_mapping
        self.constant_mapping = constant_mapping
        self.declarations = {}
        self.declarers = {}

        self.converter = LTNConverter(predicates=self.predicates, functions=self.functions, connective_impls=self.connective_impls, 
                                      quantifier_impls=self.quantifier_impls, declarations =  self.declarations, declarers = self.declarers)
        
        self.factory = ModuleFactory(converter=self.converter)

        for rule in self.ancillary_rules:
            self.factory.create_module(rule)

        self.set_rules()

        try:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        except:
            print("No parameters to optimize")

    def set_rules(self):
        self.rules = [ self.converter(rule) for rule in self.learning_rules ]
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

    def optimize(self, num_epochs=10, log_steps=10):

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