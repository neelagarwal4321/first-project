import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

input_layer_size = 12
hidden_layer1_size = 3
hidden_layer2_size = 3
output_layer_size = 2

np.random.seed(42)

W1 = np.random.randn(input_layer_size, hidden_layer1_size)
b1 = np.zeros((1, hidden_layer1_size))

W2 = np.random.randn(hidden_layer1_size, hidden_layer2_size)
b2 = np.zeros((1, hidden_layer2_size))

W3 = np.random.randn(hidden_layer2_size, output_layer_size)
b3 = np.zeros((1, output_layer_size))

# This is just passing data forward through the network
def forward_pass(X):
    z1 = sigmoid(np.dot(X,W1) + b1)
    z2 = sigmoid(np.dot(z1,W2) + b2)
    out = sigmoid(np.dot(z2,W3) + b3)
    return z1, z2, out

# Training with Backpropogation
def train(X, y, lr=0.1, epochs=5000):
    global W1, b1, W2, b2, W3, b3

    for epoch in range(epochs):

        # Forward Pass
        z1, z2, out = forward_pass(X)

        # Loss Calculation
        loss = np.mean((y - out) ** 2)

        # Backward Propogation

        # Output layer error
        d_out = (y - out) * sigmoid_derivative(out)

        # Hidden layer2 error
        d_z2 = np.dot(d_out, W3.T) * sigmoid_derivative(z2)

        # Hidden layer1 error
        d_z1 = np.dot(d_z2, W2.T) * sigmoid_derivative(z1)

        # Gradient descent updates

        W3 += lr * np.dot(z2.T, d_out)
        b3 += lr * np.sum(d_out, axis=0, keepdims=True)

        W2 += lr * np.dot(z1.T, d_z2)
        b2 += lr * np.sum(d_z2, axis=0, keepdims=True)

        W1 += lr * np.dot(X.T, d_z1)
        b1 += lr * np.sum(d_z1, axis=0, keepdims=True)

        # Print loss occasionally
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Example training
# 5 samples and totally 12 features (inputs)
X = np.array([
    [0,1,0,1,0,1,0,1,0,1,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0.5]*12,
    np.linspace(0,1,12)
])

# Target outputs (5 samples × 2 outputs)
y = np.array([
    [1,0],
    [0,1],
    [1,1],
    [0,0],
    [1,0]
])

train(X, y, lr=0.1, epochs=5000)

_, _, output = forward_pass(X)
print("\nFinal predictions:\n", output)