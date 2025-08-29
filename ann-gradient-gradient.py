import math
import random

# --- Helper functions ---
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


# --- ANN Class ---
class SimpleANN:
    def __init__(self, n_inputs, n_hidden, n_outputs):
        # Random weight initialization
        self.W1 = [[random.uniform(-1, 1) for _ in range(n_hidden)] for _ in range(n_inputs)]
        self.b1 = [0.0 for _ in range(n_hidden)]

        self.W2 = [[random.uniform(-1, 1) for _ in range(n_outputs)] for _ in range(n_hidden)]
        self.b2 = [0.0 for _ in range(n_outputs)]

    def forward(self, X):
        """Forward pass for one sample"""
        # Hidden layer
        self.z1 = [0.0 for _ in range(len(self.b1))]
        for j in range(len(self.b1)):
            self.z1[j] = sigmoid(sum(X[i] * self.W1[i][j] for i in range(len(X))) + self.b1[j])

        # Output layer
        self.out = [0.0 for _ in range(len(self.b2))]
        for k in range(len(self.b2)):
            self.out[k] = sigmoid(sum(self.z1[j] * self.W2[j][k] for j in range(len(self.z1))) + self.b2[k])

        return self.out

    def train(self, X, y, lr=0.1, epochs=5000):
        for epoch in range(epochs):
            total_loss = 0
            for xi, yi in zip(X, y):
                # ---- Forward ----
                out = self.forward(xi)

                # ---- Loss ---- (Mean Squared Error)
                errors = [yi[k] - out[k] for k in range(len(y[0]))]
                total_loss += sum(e**2 for e in errors) / len(errors)

                # ---- Backpropagation ----
                # Output layer deltas
                d_out = [errors[k] * sigmoid_derivative(out[k]) for k in range(len(out))]

                # Hidden layer deltas
                d_hidden = []
                for j in range(len(self.z1)):
                    err = sum(d_out[k] * self.W2[j][k] for k in range(len(d_out)))
                    d_hidden.append(err * sigmoid_derivative(self.z1[j]))

                # ---- Update weights ----
                # Update W2, b2
                for j in range(len(self.W2)):
                    for k in range(len(self.W2[j])):
                        self.W2[j][k] += lr * d_out[k] * self.z1[j]
                for k in range(len(self.b2)):
                    self.b2[k] += lr * d_out[k]

                # Update W1, b1
                for i in range(len(self.W1)):
                    for j in range(len(self.W1[i])):
                        self.W1[i][j] += lr * d_hidden[j] * xi[i]
                for j in range(len(self.b1)):
                    self.b1[j] += lr * d_hidden[j]

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(X):.4f}")

    def predict(self, X):
        preds = []
        for xi in X:
            preds.append(self.forward(xi))
        return preds


# --- Example Usage ---

# Simple dataset: XOR problem
X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

y = [
    [0],
    [1],
    [1],
    [0]
]

# Create ANN: 2 inputs → 3 hidden → 1 output
model = SimpleANN(n_inputs=2, n_hidden=3, n_outputs=1)

# Train
model.train(X, y, lr=0.5, epochs=5000)

# Test
print("\nPredictions:")
for xi in X:
    print(f"Input: {xi}, Predicted: {model.predict([xi])[0]}")
