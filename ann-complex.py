import math
import random

# --- Activation function ---
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# --- Artificial Neural Network ---
class NeuralNetwork:
    def __init__(self, n_inputs, n_hidden, n_hidden2, n_outputs):
        # Layer 1 (input → hidden1)
        self.W1 = [[random.uniform(-1,1) for _ in range(n_hidden)] for _ in range(n_inputs)]
        self.b1 = [0.0 for _ in range(n_hidden)]

        # Layer 2 (hidden1 → hidden2)
        self.W2 = [[random.uniform(-1,1) for _ in range(n_hidden2)] for _ in range(n_hidden)]
        self.b2 = [0.0 for _ in range(n_hidden2)]

        # Output layer (hidden2 → output)
        self.W3 = [[random.uniform(-1,1) for _ in range(n_outputs)] for _ in range(n_hidden2)]
        self.b3 = [0.0 for _ in range(n_outputs)]

    def forward(self, x):
        # Hidden layer 1
        self.z1 = [sigmoid(sum(x[i]*self.W1[i][j] for i in range(len(x))) + self.b1[j]) for j in range(len(self.b1))]
        # Hidden layer 2
        self.z2 = [sigmoid(sum(self.z1[i]*self.W2[i][j] for i in range(len(self.z1))) + self.b2[j]) for j in range(len(self.b2))]
        # Output
        self.out = [sigmoid(sum(self.z2[i]*self.W3[i][j] for i in range(len(self.z2))) + self.b3[j]) for j in range(len(self.b3))]
        return self.out

    def train(self, X, y, lr=0.1, epochs=1000):
        for epoch in range(epochs):
            total_loss = 0
            for xi, yi in zip(X,y):
                # Forward
                out = self.forward(xi)

                # Compute error
                errors = [yi[j]-out[j] for j in range(len(yi))]
                total_loss += sum(e**2 for e in errors)

                # Backpropagation (Output layer)
                d_out = [errors[j]*sigmoid_derivative(out[j]) for j in range(len(out))]

                # Backprop to hidden2
                d_hidden2 = []
                for j in range(len(self.z2)):
                    err = sum(d_out[k]*self.W3[j][k] for k in range(len(d_out)))
                    d_hidden2.append(err*sigmoid_derivative(self.z2[j]))

                # Backprop to hidden1
                d_hidden1 = []
                for j in range(len(self.z1)):
                    err = sum(d_hidden2[k]*self.W2[j][k] for k in range(len(self.z2)))
                    d_hidden1.append(err*sigmoid_derivative(self.z1[j]))

                # Update W3, b3
                for i in range(len(self.W3)):
                    for j in range(len(self.W3[i])):
                        self.W3[i][j] += lr*d_out[j]*self.z2[i]
                for j in range(len(self.b3)):
                    self.b3[j] += lr*d_out[j]

                # Update W2, b2
                for i in range(len(self.W2)):
                    for j in range(len(self.W2[i])):
                        self.W2[i][j] += lr*d_hidden2[j]*self.z1[i]
                for j in range(len(self.b2)):
                    self.b2[j] += lr*d_hidden2[j]

                # Update W1, b1
                for i in range(len(self.W1)):
                    for j in range(len(self.W1[i])):
                        self.W1[i][j] += lr*d_hidden1[j]*xi[i]
                for j in range(len(self.b1)):
                    self.b1[j] += lr*d_hidden1[j]

            if epoch % 200 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    def predict(self, X):
        return [self.forward(xi) for xi in X]


# --- Synthetic Dataset ---
# Simple XOR problem extended with noise
X = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]
y = [
    [0],
    [1],
    [1],
    [0]
]

# --- Train ANN ---
nn = NeuralNetwork(n_inputs=2, n_hidden=6, n_hidden2=6, n_outputs=1)
nn.train(X, y, lr=0.5, epochs=2000)

# --- Predictions ---
print("\nPredictions:")
for xi in X:
    print(f"Input: {xi}, Predicted: {nn.predict([xi])[0]}")
