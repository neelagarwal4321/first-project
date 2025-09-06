import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# --- Step 1: Create dataset ---
# Feature = hours studied, Target = pass(1)/fail(0)
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y = np.array([0,0,0,0,1,1,1,1,1,1])  # label

# --- Step 2: Train Logistic Regression model ---
model = LogisticRegression()
model.fit(X, y)

# --- Step 3: Make predictions ---
X_test = np.array([[2],[4],[6],[8],[10]])
y_pred = model.predict(X_test)

print("Predictions:", y_pred)

# --- Step 4: Evaluate ---
y_train_pred = model.predict(X)
print("Accuracy:", accuracy_score(y, y_train_pred))
print("Confusion Matrix:\n", confusion_matrix(y, y_train_pred))

# --- Step 5: Visualization ---
plt.scatter(X, y, color='blue', label='Data')
X_range = np.linspace(0, 10, 100).reshape(-1,1)
y_prob = model.predict_proba(X_range)[:,1]
plt.plot(X_range, y_prob, color='red', linewidth=2, label='Logistic Curve')
plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression Example")
plt.legend()
plt.show()
