import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


file_path = 'MarvellousAdvertising.csv'  
data = pd.read_csv(file_path)

X = data[['TV', 'radio', 'newspaper']]
y = data['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]  # Add a column of ones to the training set
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]     # Add a column of ones to the testing set

theta_best = np.linalg.inv(X_train_b.T.dot(X_train_b)).dot(X_train_b.T).dot(y_train)

def predict(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add a column of ones to the input data
    return X_b.dot(theta)

# Predict on the test set
# Use the trained model to make predictions on the testing set
y_pred_custom = predict(X_test, theta_best)

# Calculate the R-squared value
# This metric indicates how well the model fits the data (higher is better, with a maximum of 1)
r_squared_custom = r2_score(y_test, y_pred_custom)

# Display the coefficients and R-squared value
# Print the results to see the computed coefficients and the model's performance
print("Coefficients:", theta_best)
print("R-squared:", r_squared_custom)
