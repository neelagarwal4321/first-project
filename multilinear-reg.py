class MultipleLinearRegression:

    def __init__(self):
        self.m = 0
        self.b = 0

    def fit(self, X, y, lr=0.1, epochs=10000):
        
        n_samples = len(X)
        n_features = len(X[0])

        self.m = [0.0 for i in range (n_features)]
        self.b = 0.0

        for _ in range(epochs):
            for i in range(n_samples):
                y_pred = sum(self.m[j] * X[i][j] for j in range(n_features)) + self.b

                error = y[i] - y_pred

                for j in range(n_features):
                    self.m[j] += lr * error * X[i][j]
                self.b += lr * error


    def predict(self, X):
        
        preds = []
        for row in X:
            y_pred = sum(self.m[j] * row[j] for j in range(len(row))) + self.b
            preds.append(y_pred)
        return preds
    
# Dataset with 8 features
X = [
    [1,2,3,4,5,6,7,8],
    [2,1,0,3,2,1,4,5],
    [3,4,2,1,0,2,3,1],
    [5,2,1,0,1,3,2,4],
    [4,3,5,6,2,1,0,2]
]

# Target y based on known formula
y = []
for row in X:
    val = (1*row[0] + 2*row[1] + 3*row[2] + 4*row[3] +
           5*row[4] + 6*row[5] + 7*row[6] + 8*row[7] + 10)
    y.append(val)

# Train model
model = MultipleLinearRegression()
model.fit(X, y, lr=0.001, epochs=20000)

# Print learned coefficients
print("Weights:", [round(c, 2) for c in model.m])
print("Bias:", round(model.b, 2))

# Predictions
X_test = [
    [1,1,1,1,1,1,1,1],
    [2,2,2,2,2,2,2,2]
]
preds = model.predict(X_test)
print("Predictions:", preds)