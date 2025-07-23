import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

def visualize_data(X_train, y_train, X_test=None, y_test=None):
    """
    Visualize training data in green; test data in red if provided.
    """
    # Convert to NumPy if Torch
    X_train_np = X_train.numpy() if isinstance(X_train, torch.Tensor) else X_train
    y_train_np = y_train.numpy() if isinstance(y_train, torch.Tensor) else y_train
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train_np, y_train_np, color='green', label='Train Data')
    if X_test is not None and y_test is not None:
        X_test_np = X_test.numpy() if isinstance(X_test, torch.Tensor) else X_test
        y_test_np = y_test.numpy() if isinstance(y_test, torch.Tensor) else y_test
        plt.scatter(X_test_np, y_test_np, color='red', label='Test Data')
    plt.title('Data Visualization')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def plot_true_vs_pred(X, y_true, y_pred):
    """
    Scatter plot against X: y_true in green, y_pred in blue; vertical lines show errors.
    """
    # Convert to NumPy if Torch
    X_np = X.numpy() if isinstance(X, torch.Tensor) else X
    y_true_np = y_true.detach().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred_np = y_pred.detach().numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    
    # Sort by X for better visualization
    sort_idx = np.argsort(X_np)
    X_sorted = X_np[sort_idx]
    y_true_sorted = y_true_np[sort_idx]
    y_pred_sorted = y_pred_np[sort_idx]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_sorted, y_true_sorted, color='green', label='y_true')
    plt.scatter(X_sorted, y_pred_sorted, color='blue', label='y_pred')
    for i in range(len(X_sorted)):
        plt.plot([X_sorted[i], X_sorted[i]], [y_true_sorted[i], y_pred_sorted[i]], color='gray', linestyle='--')
    plt.title('True vs. Predicted Values with Errors (vs. X)')
    plt.xlabel('X')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def plot_loss_surface(X, y, weight_point, bias_point):
    """
    Plot 2D loss surface for weight and bias; mark the given point.
    """
    weights = torch.linspace(-10, 10, 100)
    biases = torch.linspace(-10, 10, 100)
    W, B = torch.meshgrid(weights, biases)
    losses = torch.zeros_like(W)

    criterion = nn.MSELoss()
    
    for i in range(100):
        for j in range(100):
            y_pred = W[i, j] * X + B[i, j]
            losses[i, j] = criterion(y, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(W.numpy(), B.numpy(), losses.numpy(), levels=50, cmap='viridis')
    plt.colorbar(label='MSE Loss')
    plt.scatter(weight_point, bias_point, color='red', label='Point')
    plt.title('Loss Surface for Linear Model')
    plt.xlabel('Weight')
    plt.ylabel('Bias')
    plt.legend()
    plt.show()

def generate_data(n_samples=100, noise=0.5, seed=42):
    """
    Generate synthetic data with added noise.
    Returns Torch tensors X (1D features) and y (targets).
    """
    np.random.seed(seed)
    X_np = np.random.uniform(-5, 5, n_samples)
    y_np = 0.5 * X_np**3 + X_np**2 + 2 * X_np + 1 + np.random.normal(0, noise, n_samples)
    return torch.tensor(X_np, dtype=torch.float32), torch.tensor(y_np, dtype=torch.float32)

def train_test_split(X, y, test_size=0.2, seed=None):
    """
    Split data into train and test sets. Seed is optional for reproducibility.
    """
    # Convert to NumPy for shuffling, then back to Torch
    if seed is not None:
        np.random.seed(seed)
    X_np = X.numpy() if isinstance(X, torch.Tensor) else X
    y_np = y.numpy() if isinstance(y, torch.Tensor) else y
    indices = np.arange(len(X_np))
    np.random.shuffle(indices)
    test_samples = int(len(X_np) * test_size)
    test_idx = indices[:test_samples]
    train_idx = indices[test_samples:]
    return (X[train_idx], X[test_idx], y[train_idx], y[test_idx]) if isinstance(X, torch.Tensor) else (torch.tensor(X_np[train_idx]), torch.tensor(X_np[test_idx]), torch.tensor(y_np[train_idx]), torch.tensor(y_np[test_idx]))

def create_polynomial_features(X, degree):
    """
    Create polynomial features up to the given degree.
    """
    powers = torch.arange(1, degree + 1, dtype=torch.float32)
    return torch.pow(X.unsqueeze(1), powers).squeeze() if degree > 0 else torch.ones_like(X).unsqueeze(1)