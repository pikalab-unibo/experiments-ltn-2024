import numpy as np
import pandas as pd
from scipy.stats import entropy

def apply_noise_general(data: pd.DataFrame, mu: float = 0, sigma: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """
    Apply Gaussian noise to all numeric features in the dataset in a general manner.
    
    :param data: Original dataset (DataFrame)
    :param mu: Mean of the Gaussian noise (default 0)
    :param sigma: Standard deviation of the Gaussian noise (default 0.1)
    :param seed: Random seed for reproducibility
    :return: DataFrame with noise applied to numeric features
    """
    np.random.seed(seed)
    
    # Create a copy of the data to avoid modifying the original
    noisy_data = data.copy()
    
    # Identify numeric columns
    numeric_columns = noisy_data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Apply Gaussian noise to each numeric feature
    for col in numeric_columns:
        # Apply noise and ensure values remain within feature's original range
        noisy_data[col] = noisy_data[col] + np.random.normal(mu, sigma, size=noisy_data[col].shape)
        
        # Clip values to ensure they remain reasonable based on feature type (e.g., no negative values for Age)
        if noisy_data[col].min() >= 0:  # Assuming the feature should be non-negative
            noisy_data[col] = noisy_data[col].clip(lower=0)
    
    return noisy_data

def flip_labels_general(data: pd.DataFrame, label_column: str, flip_prob: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """
    Flip the labels in the dataset with a given probability.
    
    :param data: Original dataset (DataFrame)
    :param label_column: Name of the column containing the labels
    :param flip_prob: Probability of flipping each label (default 0.1)
    :param seed: Random seed for reproducibility
    :return: DataFrame with labels flipped
    """
    np.random.seed(seed)
    
    # Create a copy of the data to avoid modifying the original
    flipped_data = data.copy()
    
    # Get unique labels in the label column and ensure they are integers
    possible_labels = flipped_data[label_column].unique().astype(int)
    
    def flip_label(label):
        if np.random.rand() < flip_prob:
            # Flip the label to any other possible label, ensuring integer output
            return int(np.random.choice([l for l in possible_labels if l != label]))
        return int(label)  # Ensure the original label remains an integer
    
    # Apply the flipping function to the label column
    flipped_data[label_column] = flipped_data[label_column].apply(flip_label)
    
    return flipped_data


def compute_kl_divergence(original_data: pd.DataFrame, perturbed_data: pd.DataFrame, feature_columns: list) -> float:
    """
    Compute the Kullback-Leibler (KL) divergence between the original and perturbed datasets.
    
    :param original_data: Original dataset (DataFrame)
    :param perturbed_data: Perturbed dataset (DataFrame)
    :param feature_columns: List of feature column names to include in the divergence calculation
    :return: KL divergence score
    """
    kl_divergence = 0.0
    n_features = len(feature_columns)

    for feature in feature_columns:
        # Get the distributions of the feature values in both original and perturbed data
        original_distribution = original_data[feature].value_counts(normalize=True).sort_index()
        perturbed_distribution = perturbed_data[feature].value_counts(normalize=True).sort_index()
        
        # Align indices to ensure the distributions are comparable
        original_distribution, perturbed_distribution = original_distribution.align(perturbed_distribution, fill_value=1e-10)
        
        # Compute KL divergence for this feature
        kl_divergence += entropy(original_distribution, perturbed_distribution)
    
    # Return average KL divergence across all features
    return kl_divergence / n_features

def calculate_absolute_and_relative_robustness(original_accuracy: float, perturbed_accuracy: float, 
                                               uneducated_accuracy: float) -> (float, float):
    """
    Calculate absolute and relative robustness of a model.
    
    :param original_accuracy: Accuracy of the model on the original dataset
    :param perturbed_accuracy: Accuracy of the model on the perturbed dataset
    :param uneducated_accuracy: Accuracy of a baseline or 'uneducated' model
    :return: Absolute robustness and relative robustness
    """
    # Absolute robustness compares original and perturbed accuracies
    absolute_robustness = perturbed_accuracy / original_accuracy if original_accuracy != 0 else 0
    
    # Relative robustness compares the absolute robustness to the uneducated model's performance
    relative_robustness = absolute_robustness / uneducated_accuracy if uneducated_accuracy != 0 else 0
    
    return absolute_robustness, relative_robustness


def apply_combined_perturbations(data: pd.DataFrame, label_column: str, mu: float = 0, sigma: float = 0.1,
                                 flip_prob: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """
    Apply both noise and label flipping to the dataset.
    
    :param data: Original dataset (DataFrame)
    :param label_column: Name of the column containing the labels
    :param mu: Mean of the Gaussian noise (default 0)
    :param sigma: Standard deviation of the Gaussian noise (default 0.1)
    :param flip_prob: Probability of flipping each label (default 0.1)
    :param seed: Random seed for reproducibility
    :return: DataFrame with both noise and label flipping applied
    """
    # Apply noise
    noisy_data = apply_noise_general(data, mu=mu, sigma=sigma, seed=seed)
    
    # Apply label flipping
    perturbed_data = flip_labels_general(noisy_data, label_column=label_column, flip_prob=flip_prob, seed=seed)
    
    return perturbed_data