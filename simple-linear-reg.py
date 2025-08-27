# Simple dataset
X = [1, 2, 3, 4, 5]
y = [5, 7, 9, 11, 13]  # because y = 2x + 3

class LinearRegression:
    def __init__(self):
        self.m = 0  # slope
        self.b = 0  # intercept

    def fit(self, X, y):
        
        mean_x = sum(X)/(len(X))
        mean_y = sum(y)/(len(X))

        num = 0
        den = 0

        for i in range(len(X)):
            num = num + ((X[i] - mean_x)*(y[i] - mean_y))
            den = den + (X[i] - mean_x) ** 2

        self.m = num/den
        self.b = mean_y - self.m * mean_x

    def predict(self, X):
        
        return [self.m * x + self.b for x in X]
    

model = LinearRegression()

model.fit(X,y)

print("Slope(m): ", model.m)
print("Intercept(b): ", model.b)

X_test = [6,7,8]
predictions = model.predict(X_test)

print("Prediction for ", X_test, ":", predictions)

