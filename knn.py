import math
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        # Store training data
        self.X_train = X
        self.y_train = y

    def distance(self, x1, x2):
        # Euclidean distance
        return math.sqrt(sum((a-b)**2 for a,b in zip(x1, x2)))

    def predict_one(self, x):
        # Compute all distances
        distances = [(self.distance(x, x_train), y) for x_train, y in zip(self.X_train, self.y_train)]
        # Sort by distance
        distances.sort(key=lambda d: d[0])
        # Take k nearest labels
        k_labels = [label for _, label in distances[:self.k]]
        # Majority vote
        return Counter(k_labels).most_common(1)[0][0]

    def predict(self, X):
        return [self.predict_one(x) for x in X]


# --- Example Dataset ---
# Features = [x, y], Labels = class
X_train = [
    [1,2], [2,3], [3,1],   # Class 0
    [6,5], [7,7], [8,6]    # Class 1
]
y_train = [0,0,0,1,1,1]

# --- Train ---
knn = KNN(k=3)
knn.fit(X_train, y_train)

# --- Predict ---
X_test = [[2,2], [7,6], [5,5]]
predictions = knn.predict(X_test)

print("Test points:", X_test)
print("Predictions:", predictions)
