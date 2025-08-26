import math
import random

# Sigmoid activation and derivative
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights and biases manually
def init_matrix(rows, cols):
    return [[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]

def init_bias(size):
    return [0.0 for _ in range(size)]

# Dot product
def dot_product(vec, mat):
    return [sum(v*m for v, m in zip(vec, col)) for col in zip(*mat)]

# Add bias
def add_bias(vec, bias):
    return [v + b for v, b in zip(vec, bias)]

# Forward pass (1 sample)
def forward_pass(x, W1, b1, W2, b2, W3, b3):
    # Layer 1
    z1_raw = add_bias(dot_product(x, W1), b1)
    z1 = [sigmoid(v) for v in z1_raw]
    # Layer 2
    z2_raw = add_bias(dot_product(z1, W2), b2)
    z2 = [sigmoid(v) for v in z2_raw]
    # Output
    out_raw = add_bias(dot_product(z2, W3), b3)
    out = [sigmoid(v) for v in out_raw]
    return z1, z2, out

# Backpropagation (update weights)
def train(X, Y, epochs=5000, lr=0.1):
    # Initialize weights and biases
    W1 = init_matrix(12, 3)
    b1 = init_bias(3)
    W2 = init_matrix(3, 3)
    b2 = init_bias(3)
    W3 = init_matrix(3, 2)
    b3 = init_bias(2)

    for epoch in range(epochs):
        total_loss = 0
        for x, y in zip(X, Y):
            # ---- Forward ----
            z1, z2, out = forward_pass(x, W1, b1, W2, b2, W3, b3)

            # ---- Loss (MSE) ----
            errors = [yt - ot for yt, ot in zip(y, out)]
            total_loss += sum(e**2 for e in errors) / len(errors)

            # ---- Backpropagation ----
            # Output layer delta
            d_out = [e * sigmoid_derivative(o) for e, o in zip(errors, out)]

            # Hidden layer 2 delta
            d_z2 = []
            for j in range(len(z2)):
                err = sum(d_out[k] * W3[j][k] for k in range(len(W3[j])))
                d_z2.append(err * sigmoid_derivative(z2[j]))

            # Hidden layer 1 delta
            d_z1 = []
            for j in range(len(z1)):
                err = sum(d_z2[k] * W2[j][k] for k in range(len(W2[j])))
                d_z1.append(err * sigmoid_derivative(z1[j]))

            # ---- Update weights & biases ----
            # W3, b3
            for j in range(len(W3)):
                for k in range(len(W3[j])):
                    W3[j][k] += lr * d_out[k] * z2[j]
            for k in range(len(b3)):
                b3[k] += lr * d_out[k]

            # W2, b2
            for j in range(len(W2)):
                for k in range(len(W2[j])):
                    W2[j][k] += lr * d_z2[k] * z1[j]
            for k in range(len(b2)):
                b2[k] += lr * d_z2[k]

            # W1, b1
            for j in range(len(W1)):
                for k in range(len(W1[j])):
                    W1[j][k] += lr * d_z1[k] * x[j]
            for k in range(len(b1)):
                b1[k] += lr * d_z1[k]

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(X):.4f}")

    return W1, b1, W2, b2, W3, b3

# ---- Example Training ----
X = [
    [0,1,0,1,0,1,0,1,0,1,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0.5]*12,
    [i/11 for i in range(12)]
]

Y = [
    [1,0],
    [0,1],
    [1,1],
    [0,0],
    [1,0]
]

# Train
W1, b1, W2, b2, W3, b3 = train(X, Y, epochs=5000, lr=0.1)

# Test
for x in X:
    _, _, out = forward_pass(x, W1, b1, W2, b2, W3, b3)
    print("Input:", x, "Output:", [round(v,3) for v in out])
