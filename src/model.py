import numpy as np
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

def train(X, y, lr=0.05, epochs=500):
    """Train logistic regression model using gradient descent."""
    Xb = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(epochs):
        w -= lr * Xb.T @ (sigmoid(Xb @ w) - y) / len(y)
    return w

def score(X, w):
    """Generate fraud probability scores."""
    Xb = np.column_stack([np.ones(len(X)), X])
    return sigmoid(Xb @ w)

def normalize(X):
    """Normalize features to zero mean and unit variance."""
    m, s = X.mean(0), X.std(0) + 1e-8
    return (X - m) / s, m, s

def importance(w, names):
    """Return feature importance ranked."""
    return pd.Series(np.abs(w[1:]), index=names).sort_values(ascending=False)
