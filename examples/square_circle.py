# Data Processing
import pandas as pd
from PIL import Image
from examples.generator import generate_balanced_dataset
import numpy as np
from sklearn.model_selection import train_test_split

# Torch
import torch
from torch.utils.data import DataLoader, Dataset

# LTN
from ltn_imp.automation.knowledge_base import KnowledgeBase
from ltn_imp.automation.data_loaders import LoaderWrapper

from models import *

data = pd.DataFrame(generate_balanced_dataset(100))

image_paths = [item for item in data[0]]
images = []

for path in image_paths:
    try:
        img = Image.open(path).convert('RGB')  # Convert to RGB to ensure consistency
        img = np.array(img)
        img_tensor = torch.tensor(img, dtype=torch.float32)  # Convert to PyTorch tensor
        images.append(img_tensor)
    except Exception as e:
        print(f"Error loading image {path}: {e}")

labels = torch.tensor(data[1], dtype=torch.float32)

class ImageDataset(Dataset):
    def __init__(self, images, labels):
        # Convert the list of images to a tensor and permute dimensions to [batch_size, channels, height, width]
        self.images = torch.stack([image.clone().detach().permute(2, 0, 1) for image in images])
        self.labels = labels.clone().detach().float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

batch_size = 64

# Split the data into training and test sets
train_data, test_data, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create the training dataset
train_dataset = ImageDataset(train_data, train_labels)

# Create the test dataset
test_dataset = ImageDataset(test_data, test_labels)

# Create the training dataloader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create the test dataloader
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

ancillary_rules = [
    "forall c1 c2 r b11 b12 t11 t12. Inside(c1, c2, r, b11, b12, t11, t12) <-> (((b11 + r) <= c1 and (c1 <= (t11 - r))) and ((b12 + r) <= c2 and (c2 <= (t12 - r))))",
    "forall c1 c2 r b11 b12 t11 t12. Outside(c1, c2, r, b11, b12, t11, t12) <-> ((((c1 + r) <= b11) or ((c1 - r) >= t11) ) or ( ((c2 + r) <= b12) or ((c2 - r) >= t12)))",
    "forall c1 c2 r b11 b12 t11 t12. Intersect(c1, c2, r, b11, b12, t11, t12) <-> (not Inside(c1, c2, r, b11, b12, t11, t12) and not Outside(c1, c2, r, b11, b12, t11, t12))",
]

learning_rules = [
    "all i. ((y = in) -> (Circle(i, c1, c2, r) and Rect(i, t11, t12, b11, b12) and Inside(c1, c2, r, t11, t12, b11, b12)))",
    "all i. ((y = int) -> (Circle(i, c1, c2, r) and Rect(i, t11, t12, b11, b12) and Intersect(c1, c2, r, t11, t12, b11, b12)))",
    "all i. ((y = out) -> (Circle(i, c1, c2, r) and Rect(i, t11, t12, b11, b12) and Outside(c1, c2, r, t11, t12, b11, b12)))"
]

circle = CircleDetector()
rectangle = RectangleDetector()

predicates = {
    "Circle": circle,
    "Rect": rectangle,
}

loader = LoaderWrapper(loader=train_dataloader, variables=["i"], targets=["y"])

rule_to_loader = {rule: [loader] for rule in learning_rules }

quantifier_imp = {"forall" : "pmean_error"}

connective_imp = {"eq": "tan"}

constants = {
    "in" : torch.tensor([0.]),
    "int" : torch.tensor([1.]),
    "out" : torch.tensor([2.]),
}

kb = KnowledgeBase(
    predicates=predicates,
    ancillary_rules=ancillary_rules,
    learning_rules=learning_rules,
    rule_to_data_loader_mapping=rule_to_loader,
    quantifier_impls=quantifier_imp,
    connective_impls=connective_imp,
    constant_mapping=constants
)


print("Before Training", compute_accuracy(kb, test_dataloader, torch.device("cpu")))
print()

kb.optimize(num_epochs=5, lr=0.0001, log_steps=1)

print()
print("Before Training", compute_accuracy(kb, test_dataloader, torch.device("cpu")))
