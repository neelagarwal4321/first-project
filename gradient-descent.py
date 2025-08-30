# Simple Gradient Descent Example
import random

# Dataset: y = 2x + 1
X = [1, 2, 3, 4, 5]
y = [2*x + 1 for x in X]

# Initialize parameters (slope and intercept)
w = random.random()   # slope (start random)
b = random.random()   # intercept (start random)
lr = 0.01             # learning rate
epochs = 1000         # number of iterations

# Training loop
for epoch in range(epochs):
    dw, db = 0, 0
    n = len(X)

    # Calculate gradients
    for xi, yi in zip(X, y):
        y_pred = w*xi + b
        error = yi - y_pred
        dw += -2*xi*error
        db += -2*error

    dw /= n
    db /= n

    # Update weights
    w -= lr * dw
    b -= lr * db

    # Print detailed progress occasionally
    if epoch % 200 == 0:
        loss = sum((yi - (w*xi+b))**2 for xi,yi in zip(X,y)) / n
        print(f"Epoch {epoch}:")
        print(f"   Current slope (w): {w:.4f}")
        print(f"   Current intercept (b): {b:.4f}")
        print(f"   Current loss (MSE): {loss:.6f}")
        print("   → The model is trying to fit closer to the true line y=2x+1\n")

# Final model
print("\nFinal Learned Line: y = {:.2f}x + {:.2f}".format(w, b))

# Predictions
X_test = [6, 7]
y_pred = [w*xi + b for xi in X_test]
print("Predictions for", X_test, ":", y_pred)
