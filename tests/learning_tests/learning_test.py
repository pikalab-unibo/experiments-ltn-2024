import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_classification
from ltn_imp.fuzzy_operators.aggregators import AvgSatAgg
from ltn_imp.fuzzy_operators.predicates import Predicate
from ltn_imp.fuzzy_operators.connectives import NotOperation
from ltn_imp.fuzzy_operators.quantifiers import ForallQuantifier
from ltn_imp.parsing.parser import convert_to_ltn
from nltk.sem.logic import Expression

class BinaryClassificationModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BinaryClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Binary Classification Dataset Class
class BinaryClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.labels = y
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Generate binary classification dataset
num_samples = 100 
input_size = 10
hidden_size = 5

X_binary, y_binary = make_classification(n_samples=num_samples, n_features=input_size, n_classes=2, random_state=42)
y_binary = y_binary.reshape(-1, 1)  # Reshape to (100, 1)

# Convert to PyTorch tensors
X_binary = torch.tensor(X_binary, dtype=torch.float32)
y_binary = torch.tensor(y_binary, dtype=torch.float32)

# Create dataset and dataloader
binary_dataset = BinaryClassificationDataset(X_binary, y_binary)
train_loader = DataLoader(binary_dataset, batch_size=10, shuffle=True)

class TestLearning(unittest.TestCase):
    
    def setUp(self):
        input_size = 10
        hidden_size = 5

        self.model = BinaryClassificationModel(input_size,hidden_size)
        predicates = {"Classifier":self.model }


        expression_1 = Expression.fromstring("all x. (Classifier(x))")
        self.rule_1 = convert_to_ltn(expression_1, predicates=predicates, functions=None)

        expression_2 = Expression.fromstring("all x. (not Classifier(x))")
        self.rule_2 = convert_to_ltn(expression_2, predicates=predicates, functions=None)

        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.sat_agg = AvgSatAgg()

        # Use the pre-defined dataset and dataloader
        self.train_loader = train_loader

    def test_binary_classification(self):

        # Initialize previous_loss by running one forward pass
        data, labels = next(iter(self.train_loader))
        outputs = self.model(data)
        previous_loss = torch.nn.functional.binary_cross_entropy(outputs, labels).item()
        
        for epoch in range(10):
            train_loss = 0.0
            for _, (data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()

                pos = data[labels.squeeze() == 1]
                neg = data[labels.squeeze() == 0]
                

                if len(pos) == 0 or len(neg) == 0:
                    continue

                # Compute satisfaction level
                sat_agg_value = self.sat_agg(
                    self.rule_1( {"x" : pos} ),
                    self.rule_2( { "x": neg})
                )
                
                # Compute loss
                loss = 1.0 - sat_agg_value
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(self.train_loader)

            # Check if the loss is decreasing
        self.assertLess(train_loss, previous_loss, "Loss did not decrease after training")


if __name__ == '__main__':
    unittest.main(buffer=False)
