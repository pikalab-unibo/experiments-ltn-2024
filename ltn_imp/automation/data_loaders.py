from itertools import cycle


class LoaderWrapper:
    def __init__(self, variables, targets, loader):
        self.variables = variables
        self.targets = targets
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

        if loaders != []:
            self.max_length = max(len(loader) for loader in loaders)
        else:
            self.max_length = 1

    def __iter__(self):
        return self

    def __len__(self):
        return self.max_length
    
    def __next__(self):
        return self.current_batches

    def step(self):
        for loader in self.loaders:
            self.current_batches[loader] = next(self.iters[loader])


