import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --- Step 1: Create dataset ---
# X = Years of Experience, y = Salary
X = np.array([[i] for i in range(1, 21)])   # 20 samples
y = np.array([25000 + 5000*i + np.random.randint(-2000, 2000) for i in range(1, 21)])  # Add noise

# --- Step 2: Split into train and test sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 3: Train Linear Regression model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Step 4: Make predictions ---
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# --- Step 5: Evaluate ---
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("Coefficient (slope):", model.coef_[0])
print("Intercept:", model.intercept_)
print("Training MSE:", mse_train)
print("Test MSE:", mse_test)
print("Training R²:", r2_train)
print("Test R²:", r2_test)

# --- Step 6: Visualization ---
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression: Experience vs Salary")
plt.legend()
plt.show()

# --- Step 7: Predict new values ---
X_new = np.array([[21], [22], [25]])
y_new_pred = model.predict(X_new)
print("Predictions for 21, 22, 25 years:", y_new_pred)
