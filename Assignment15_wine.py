import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('WinePredictor.csv')

# Step 2: Clean, Prepare, and Manipulate Data
# Check for missing values
if data.isnull().sum().any():
    data = data.dropna()

# Separate features and target variable
X = data.iloc[:, 1:].values  # Features
y = data.iloc[:, 0].values   # Target variable (Class labels)

# Normalize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train Data
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Create KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Step 4: Test Data
# Predict the class for test data
y_pred = knn.predict(X_test)

# Display the predictions
print(y_pred)

# Step 5: Calculate Accuracy
def CheckAccuracy(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Calculate accuracy
accuracy = CheckAccuracy(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Calculate accuracy for different values of k
def CheckAccuracyForKValues(X_train, X_test, y_train, y_test, k_values):
    accuracies = {}
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[k] = accuracy
    return accuracies

# Test accuracy for different values of k
k_values = [1, 3, 5, 7, 9]
accuracies = CheckAccuracyForKValues(X_train, X_test, y_train, y_test, k_values)
print(accuracies)
