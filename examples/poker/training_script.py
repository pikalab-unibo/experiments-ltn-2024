from ltn_imp.automation.knowledge_base import KnowledgeBase

from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import os
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")

output_dir = os.getenv("OUTPUT_DIR", "./output")
lr = 0.016817041194092816
weight_decay = 0.00248758573986529
epochs = 25
num_seeds = 30

def prepare_datasets(data, random_seed=42):
    X = data.drop("Label", axis=1)  # Features
    y = data["Label"]  # Labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_seed, stratify=y_train)

    train_data = pd.concat([X_train, y_train], axis=1)
    val_data = pd.concat([X_val, y_val], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_data.to_csv("datasets/train.csv", index=False)
    val_data.to_csv("datasets/val.csv", index=False)
    test_data.to_csv("datasets/test.csv", index=False)

    return data, train_data, val_data, test_data

def find_best_models(X,y):
    param_grid = {
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            }
        },
        'DT': {
            'model': DecisionTreeClassifier(),
            'params': {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            }
        },
        'RF': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'criterion': ['gini', 'entropy']
            }
        },
        'LR': {
            'model': LogisticRegression(max_iter=1000),
            'params': {
                'C': [0.1, 1, 10],
                'solver': ['lbfgs', 'liblinear']
            }
        }
    }

    best_models = {}
    for name, model_info in param_grid.items():
        grid_search = GridSearchCV(model_info['model'], model_info['params'], cv=5, n_jobs=-1)
        grid_search.fit(X, y)
        best_models[name] = grid_search.best_estimator_

    models = []
    for name, model in best_models.items():
        if name == 'KNN':
            models.append(KNeighborsClassifier(**model.get_params()))
        elif name == 'DT':
            models.append(DecisionTreeClassifier(**model.get_params()))
        elif name == 'RF':
            models.append(RandomForestClassifier(**model.get_params()))
        elif name == 'LR':
            models.append(LogisticRegression(**model.get_params()))

    return models

def evaluate_model(loader, model, device):
    all_labels = []
    all_predictions = []

    # Iterate over the data loader
    for data, labels in loader:
        # Move data and labels to the specified device
        data = data.to(device)
        labels = labels.to(device)

        # Get predictions from the model
        if isinstance(model, torch.nn.Module):            
            with torch.no_grad():
                predictions = model(data)
                predicted_labels = torch.argmax(predictions, dim=1)
        else:
            predicted_labels = model.predict(data)

    if isinstance(model, torch.nn.Module):
        all_labels.extend(labels.cpu().numpy())  
        all_predictions.extend(predicted_labels.cpu().numpy()) 
    else:
        all_labels.extend(labels)
        all_predictions.extend(predicted_labels)

    if isinstance(model, torch.nn.Module):
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

    # Compute sklearn metrics
    overall_accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro') 
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    return overall_accuracy, precision, recall, f1

def train_model(model, train_loader, device, max_epochs=epochs):

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)

    # Loop over epochs
    for _ in range(max_epochs):
        model.train()

        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            labels = labels.view(-1).long()
            
            optimizer.zero_grad()
            outputs = model(data)            
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
    return model

poker_hand = fetch_ucirepo(id=158)   
X = poker_hand.data.features 
y = poker_hand.data.targets 
data = pd.DataFrame(X, columns=poker_hand.data.feature_names)
data["Label"] = y

data, train, val, test = prepare_datasets(data)
models = find_best_models(train.drop("Label", axis=1), train["Label"])

kb = KnowledgeBase("config.yaml")
models.append(kb.predicates["PokerHand"])

model_names = [type(model).__name__ for model in models]
model_names.remove("Sequential")
model_names.append("Regular MLP")
model_names.append("SKI MLP")

metrics = ["Overall Accuracy", "Precision", "Recall", "F1"]
metrics_df = pd.DataFrame([ [ [] for _ in metrics ] for _ in model_names ] , columns=metrics, index=[model_names])

seeds = [seed for seed in range(0,num_seeds)]

for seed in seeds:
    data, train, val, test = prepare_datasets(data, seed)

    for model in models:
        model_name = type(model).__name__
        if isinstance(model, torch.nn.Module):
            kb = KnowledgeBase("config.yaml")
            model = copy.deepcopy( kb.predicates["PokerHand"] ) 
            train_model(model, kb.loaders[0], kb.device)

            metrics_values = evaluate_model(kb.test_loaders[0], model, kb.device)
            for metric, value in zip(metrics, metrics_values):
                metrics_df.loc["Regular MLP"][metric][0].append(value)
            
            kb.optimize(num_epochs=epochs, lr=lr, early_stopping=True, verbose=False)
            model_name = "SKI MLP"
            model =  kb.predicates["PokerHand"] 
        else:
            model.fit(train.drop("Label", axis=1), train["Label"])

        metrics_values = evaluate_model(kb.test_loaders[0], model, kb.device)
        for metric, value in zip(metrics, metrics_values):
            metrics_df.loc[model_name][metric][0].append(value)

    metrics_df.to_csv(f"{output_dir}/{seed}_results.csv")

