import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Data
data = pd.read_csv('MarvellousInfosystems_PlayPredictors.csv')


label_encoder = LabelEncoder()
data['Wether'] = label_encoder.fit_transform(data['Wether'])
data['Temperature'] = label_encoder.fit_transform(data['Temperature'])
data['Play'] = label_encoder.fit_transform(data['Play'])
print("Data After Encoding:")
print(data.head())


X = data[['Wether', 'Temperature']]
y = data['Play']

#Train the KNN Model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Test the Model with an example
test_values = [[0, 1]]  # Example test values: Wether=0, Temperature=1
prediction = knn.predict(test_values)
result = 'Yes' if prediction[0] == 1 else 'No'
print(f"\nTest Result for Wether=0, Temperature=1: {result}")

# Function to Calculate Accuracy for different K values
def check_accuracy(k):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Print accuracy for K values from 1 to 10
print("\nAccuracy for different values of K:")
for k in range(1, 11):
    accuracy = check_accuracy(k)
    print(f'Accuracy for K={k}: {accuracy:.2f}')
