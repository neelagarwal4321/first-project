class RidgeRegression:
    def __init__(self, lam=0.1):
        self.lam = lam   # regularization strength
        self.w1 = 0      # weight for feature 1
        self.w2 = 0      # weight for feature 2
        self.b = 0       # bias

    def fit(self, X, y):
        # For simplicity: use formula for 2 features with ridge
        n = len(X)

        # Means
        x1_mean = sum(row[0] for row in X) / n
        x2_mean = sum(row[1] for row in X) / n
        y_mean  = sum(y) / n

        # Variances and covariances
        var_x1 = sum((row[0] - x1_mean)**2 for row in X) / n + self.lam
        var_x2 = sum((row[1] - x2_mean)**2 for row in X) / n + self.lam
        cov_x1y = sum((row[0]-x1_mean)*(yi-y_mean) for row,yi in zip(X,y)) / n
        cov_x2y = sum((row[1]-x2_mean)*(yi-y_mean) for row,yi in zip(X,y)) / n

        # Weights (treating features independently here for simplicity)
        self.w1 = cov_x1y / var_x1
        self.w2 = cov_x2y / var_x2

        # Bias
        self.b = y_mean - (self.w1*x1_mean + self.w2*x2_mean)

    def predict(self, X):
        return [self.b + self.w1*row[0] + self.w2*row[1] for row in X]


# --- Example usage ---
# True function: y = 2*x1 + 3*x2 + 5
X = [[1,2], [2,1], [3,4], [4,3]]
y = [2*row[0] + 3*row[1] + 5 for row in X]

# Train
model = RidgeRegression(lam=0.5)
model.fit(X, y)

print("Weights:", model.w1, model.w2)
print("Bias:", model.b)

# Predict
X_test = [[5,2], [1,1]]
print("Predictions:", model.predict(X_test))
