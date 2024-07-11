from itertools import cycle


class LoaderWrapper:
    def __init__(self, variables, num_classes, target, loader):
        self.variables = variables
        self.num_classes = num_classes
        self.target = target
        self.loader = loader 

    def __iter__(self):
        return self
    
    def __next__(self):
        return next(iter(self.loader))
    
    def __len__(self):
        return len(self.loader)
    
class CombinedDataLoader:
    def __init__(self, loaders):
        self.loaders = loaders
        self.iters = {loader: cycle(loader) for loader in loaders}
        self.current_batches = {loader: None for loader in loaders}
        self.step()
        self.max_length = max(len(loader) for loader in loaders)

    def __iter__(self):
        return self

    def __len__(self):
        return self.max_length
    
    def __next__(self):
        return self.current_batches

    def step(self):
        for loader in self.loaders:
            self.current_batches[loader] = next(self.iters[loader])


