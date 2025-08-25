import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define network architecture
input_layer_size = 12
hidden_layer1_size = 3
hidden_layer2_size = 3
output_layer_size = 2

# Initialize weights randomly
np.random.seed(42)
W1 = np.random.randn(input_layer_size, hidden_layer1_size)
W2 = np.random.randn(hidden_layer1_size, hidden_layer2_size)
W3 = np.random.randn(hidden_layer2_size, output_layer_size)

# Forward pass
def forward_pass(X):
    z1 = sigmoid(np.dot(X, W1))        # First hidden layer
    z2 = sigmoid(np.dot(z1, W2))       # Second hidden layer
    output = sigmoid(np.dot(z2, W3))   # Output layer
    return output

# ---- Sample Inputs (5 samples × 12 features each) ----
X = np.array([
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],   # alternating pattern
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],   # all ones
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # all zeros
    [0.5]*12,                               # all 0.5
    np.linspace(0, 1, 12)                   # increasing sequence 0 → 1
])

# Run forward pass
output = forward_pass(X)
print("Input:\n", X)
print("\nNetwork Output:\n", output)