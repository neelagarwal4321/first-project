class PolynomialRegression:
    def __init__(self, degree=2):

        self.degree = degree
        self.weights = []
        self.bias = 0

    def transform(self, X):

        """Expand input features into polynomial terms."""
        return [[x**d for d in range(1, self.degree+1)] for x in X]

    def fit(self, X, y, lr=0.001, epochs=10000):

        X_poly = self.transform(X)
        n_samples = len(X_poly)
        n_features = len(X_poly[0])
        self.weights = [0.0] * n_features

        for _ in range(epochs):
            for i in range(n_samples):

                # Prediction
                y_pred = sum(self.weights[j] * X_poly[i][j] for j in range(n_features)) + self.bias
                error = y[i] - y_pred

                # Update
                for j in range(n_features):
                    self.weights[j] += lr * error * X_poly[i][j]
                self.bias += lr * error

    def predict(self, X):

        X_poly = self.transform(X)

        preds = []

        for row in X_poly:

            y_pred = sum(self.weights[j] * row[j] for j in range(len(row))) + self.bias
            preds.append(y_pred)

        return preds

# Dataset (x, y)
X = [1, 2, 3, 4, 5]
y = [2*x**2 + 3*x + 5 for x in X]  # true function

# Train model
model = PolynomialRegression(degree=2)
model.fit(X, y, lr=0.001, epochs=20000)

print("Weights:", [round(w,2) for w in model.weights])
print("Bias:", round(model.bias,2))

# Predictions
X_test = [6, 7]
print("Predictions:", model.predict(X_test))