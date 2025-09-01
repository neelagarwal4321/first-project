import math

# --- Helper functions ---
def gini_impurity(y):
    """Calculate Gini Impurity of a list of labels"""
    classes = set(y)
    impurity = 1
    for c in classes:
        p = y.count(c) / len(y)
        impurity -= p**2
    return impurity

def split_dataset(X, y, feature, threshold):
    """Split dataset into left/right based on feature threshold"""
    left_X, left_y, right_X, right_y = [], [], [], []
    for xi, yi in zip(X, y):
        if xi[feature] <= threshold:
            left_X.append(xi)
            left_y.append(yi)
        else:
            right_X.append(xi)
            right_y.append(yi)
    return left_X, left_y, right_X, right_y


# --- Decision Tree Node ---
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature        # which feature index to split
        self.threshold = threshold    # value to split on
        self.left = left              # left child
        self.right = right            # right child
        self.value = value            # final class if leaf node


# --- Decision Tree Class ---
class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None

    def build_tree(self, X, y, depth=0):
        # If all labels are same -> leaf node
        if len(set(y)) == 1:
            return Node(value=y[0])

        # If max depth reached -> return majority class
        if depth >= self.max_depth:
            return Node(value=max(set(y), key=y.count))

        n_features = len(X[0])
        best_gain = 0
        best_split = None

        current_impurity = gini_impurity(y)

        # Try all features and thresholds
        for feature in range(n_features):
            thresholds = set(x[feature] for x in X)
            for t in thresholds:
                left_X, left_y, right_X, right_y = split_dataset(X, y, feature, t)
                if not left_y or not right_y:
                    continue

                # Calculate Information Gain
                p = len(left_y) / len(y)
                gain = current_impurity - (p*gini_impurity(left_y) + (1-p)*gini_impurity(right_y))

                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, t, left_X, left_y, right_X, right_y)

        if best_gain == 0:
            # No good split -> majority class
            return Node(value=max(set(y), key=y.count))

        # Recurse for left and right
        feature, t, left_X, left_y, right_X, right_y = best_split
        left_child = self.build_tree(left_X, left_y, depth+1)
        right_child = self.build_tree(right_X, right_y, depth+1)
        return Node(feature, t, left_child, right_child)

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)

    def predict(self, X):
        return [self.predict_one(x, self.root) for x in X]


# --- Example Usage ---
# Simple dataset: [feature1, feature2], labels
X = [[2, 3], [1, 1], [3, 2], [6, 5], [7, 8], [8, 6]]
y = ["A", "A", "A", "B", "B", "B"]

tree = DecisionTree(max_depth=2)
tree.fit(X, y)

print("Predictions:")
for xi in X:
    print(xi, "->", tree.predict([xi])[0])
