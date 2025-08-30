class LassoRegression:
    def __init__(self, lam=0.1, lr=0.01, epochs=1000):
        self.lam = lam     # regularization strength
        self.lr = lr       # learning rate
        self.epochs = epochs
        self.w1 = 0
        self.w2 = 0
        self.b = 0

    def fit(self, X, y):
        n = len(X)

        for _ in range(self.epochs):
            dw1, dw2, db = 0, 0, 0

            for xi, yi in zip(X, y):
                y_pred = self.b + self.w1*xi[0] + self.w2*xi[1]
                error = yi - y_pred

                dw1 += -2*xi[0]*error
                dw2 += -2*xi[1]*error
                db  += -2*error

            # Average gradients
            dw1 /= n
            dw2 /= n
            db  /= n

            # L1 penalty (soft-thresholding)
            if self.w1 > 0:
                dw1 += self.lam
            elif self.w1 < 0:
                dw1 -= self.lam

            if self.w2 > 0:
                dw2 += self.lam
            elif self.w2 < 0:
                dw2 -= self.lam

            # Update weights
            self.w1 -= self.lr * dw1
            self.w2 -= self.lr * dw2
            self.b  -= self.lr * db

    def predict(self, X):
        return [self.b + self.w1*row[0] + self.w2*row[1] for row in X]


# --- Example usage ---
# True function: y = 2*x1 + 3*x2 + 5
X = [[1,2], [2,1], [3,4], [4,3]]
y = [2*row[0] + 3*row[1] + 5 for row in X]

# Train Lasso
model = LassoRegression(lam=0.1, lr=0.01, epochs=5000)
model.fit(X, y)

print("Weights:", model.w1, model.w2)
print("Bias:", model.b)

# Predict
X_test = [[5,2], [1,1]]
print("Predictions:", model.predict(X_test))
