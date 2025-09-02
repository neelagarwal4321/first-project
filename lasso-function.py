def lasso_regression(X, y, lr=0.01, lam=0.1, epochs=1000):
    # Initialize weights and bias
    w1, w2, b = 0.0, 0.0, 0.0
    n = len(X)

    for _ in range(epochs):
        dw1, dw2, db = 0, 0, 0

        for xi, yi in zip(X, y):
            y_pred = w1*xi[0] + w2*xi[1] + b
            error = yi - y_pred

            dw1 += -2*xi[0]*error
            dw2 += -2*xi[1]*error
            db  += -2*error

        # Average gradients
        dw1 /= n
        dw2 /= n
        db  /= n

        # L1 penalty (soft-thresholding style)
        if w1 > 0: dw1 += lam
        elif w1 < 0: dw1 -= lam
        if w2 > 0: dw2 += lam
        elif w2 < 0: dw2 -= lam

        # Update weights
        w1 -= lr * dw1
        w2 -= lr * dw2
        b  -= lr * db

    return w1, w2, b


# --- Example Usage ---
# True function: y = 2*x1 + 3*x2 + 5
X = [[1,2], [2,1], [3,4], [4,3]]
y = [2*row[0] + 3*row[1] + 5 for row in X]

w1, w2, b = lasso_regression(X, y, lr=0.01, lam=0.1, epochs=5000)

print("Learned weights:", w1, w2)
print("Learned bias:", b)

# Predict new values
X_test = [[5,2], [1,1]]
preds = [w1*row[0] + w2*row[1] + b for row in X_test]
print("Predictions:", preds)
