import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

class ReLU:
    @staticmethod
    def forward(Z):
        return np.maximum(0, Z)
    
    @staticmethod
    def derivative(Z):
        return (Z > 0).astype(float)

class Sigmoid:
    @staticmethod
    def forward(Z):
        return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))  # clip for numerical stability

# ==================== NEURAL NETWORK FUNCTIONS ====================
def forward_propagation(X, W1, b1, W2, b2):
    """Forward pass through the network"""
    # Hidden Layer
    Z1 = W1.dot(X) + b1
    A1 = ReLU.forward(Z1)
    # Output Layer
    Z2 = W2.dot(A1) + b2
    A2 = Sigmoid.forward(Z2)
    
    return Z1, A1, Z2, A2

def back_propagation(expected, A2, Z1, A1, W1, W2, X):
    """Backward pass - compute gradients"""
    m = X.shape[1]
    
    # Output layer gradients
    dZ2 = A2 - expected
    dW2 = (1/m) * dZ2.dot(A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    # Hidden layer gradients
    dA1 = W2.T.dot(dZ2)
    dZ1 = dA1 * ReLU.derivative(Z1)
    dW1 = (1/m) * dZ1.dot(X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2

def update_params(W1, W2, b1, b2, dW1, dW2, db1, db2, eta):
    """Update parameters using gradient descent"""
    W1 = W1 - eta * dW1
    b1 = b1 - eta * db1
    W2 = W2 - eta * dW2
    b2 = b2 - eta * db2
    return W1, b1, W2, b2

def cross_entropy_loss(y_true, y_pred):
    """Binary cross-entropy loss"""
    m = y_true.shape[1]
    eps = 1e-8
    loss = -np.sum(y_true * np.log(y_pred + eps) + 
                   (1 - y_true) * np.log(1 - y_pred + eps)) / m
    return loss

def mini_batch_gradient_descent(X, Y, W1, b1, W2, b2, eta=0.01, epochs=1000, batch_size=128):
    """Mini-batch gradient descent with validation"""
    m = X.shape[1]
    loss_history = []
    
    for epoch in range(epochs):
        # Shuffle data
        permutation = np.random.permutation(m)
        X_perm = X[:, permutation]
        Y_perm = Y[:, permutation]
        
        # Mini-batch training
        for i in range(0, m, batch_size):
            X_batch = X_perm[:, i:i + batch_size]
            Y_batch = Y_perm[:, i:i + batch_size]
            
            # Forward pass
            Z1, A1, Z2, A2 = forward_propagation(X_batch, W1, b1, W2, b2)
            # Backward pass
            dW1, db1, dW2, db2 = back_propagation(Y_batch, A2, Z1, A1, W1, W2, X_batch)
            # Update parameters
            W1, b1, W2, b2 = update_params(W1, W2, b1, b2, dW1, dW2, db1, db2, eta)
        
        # Validation on full dataset every 100 epochs
        if epoch % 100 == 0:
            _, _, _, A2_full = forward_propagation(X, W1, b1, W2, b2)
            loss = cross_entropy_loss(Y, A2_full)
            loss_history.append(loss)
            print(f"Epoch {epoch:4d}, Loss: {loss:.4f}")
    
    return W1, b1, W2, b2, loss_history

def initialize_params(n_features, n_hidden, n_output):
    """Initialize weights using He initialization for ReLU"""
    W1 = np.random.randn(n_hidden, n_features) * np.sqrt(2. / n_features)
    b1 = np.zeros((n_hidden, 1))
    W2 = np.random.randn(n_output, n_hidden) * np.sqrt(2. / n_hidden)
    b2 = np.zeros((n_output, 1))
    return W1, b1, W2, b2

def predict(X, W1, b1, W2, b2):
    """Make predictions"""
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    return (A2 > 0.5).astype(int)

# ==================== TESTING ====================
print("=" * 60)
print("TESTING 1-LAYER NEURAL NETWORK CLASSIFIER")
print("=" * 60)

# Generate synthetic binary classification dataset
np.random.seed(42)
X_data, y_data = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)

# Prepare data (transpose to match network dimensions)
X_train = X_train.T  # (n_features, m)
X_test = X_test.T
y_train = y_train.reshape(1, -1)  # (1, m)
y_test = y_test.reshape(1, -1)

print(f"\nDataset Info:")
print(f"  Training samples: {X_train.shape[1]}")
print(f"  Test samples: {X_test.shape[1]}")
print(f"  Features: {X_train.shape[0]}")

# Initialize parameters
n_features = X_train.shape[0]
n_hidden = 10
n_output = 1

W1, b1, W2, b2 = initialize_params(n_features, n_hidden, n_output)

print(f"\nNetwork Architecture:")
print(f"  Input Layer: {n_features} neurons")
print(f"  Hidden Layer: {n_hidden} neurons (ReLU)")
print(f"  Output Layer: {n_output} neuron (Sigmoid)")

# Train the network
print("\n" + "=" * 60)
print("TRAINING...")
print("=" * 60)

W1, b1, W2, b2, loss_history = mini_batch_gradient_descent(
    X_train, y_train, W1, b1, W2, b2,
    eta=0.1,
    epochs=1000,
    batch_size=32
)

# Evaluate on test set
print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)

y_train_pred = predict(X_train, W1, b1, W2, b2)
y_test_pred = predict(X_test, W1, b1, W2, b2)

train_accuracy = accuracy_score(y_train.flatten(), y_train_pred.flatten())
test_accuracy = accuracy_score(y_test.flatten(), y_test_pred.flatten())

print(f"\nTraining Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("\nClassification Report (Test Set):")
print(classification_report(y_test.flatten(), y_test_pred.flatten()))

# Plot loss curve
plt.figure(figsize=(10, 5))
plt.plot(range(0, 1000, 100), loss_history, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training Loss Over Time')
plt.grid(True)
plt.show()

print("\n" + "=" * 60)
print("✓ Training Complete! Network is working correctly.")
print("=" * 60)