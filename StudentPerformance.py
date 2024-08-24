#################################################
# Import all required modules
#################################################
from sklearn.tree import DecisionTreeClassifier
import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble  import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from matplotlib import pyplot as plt

#################################################
# Function_Name: GetData()
# Discription: Given fuction used to store data from csv file in variable
# Input: CSV file
# Output: Stores data in variable called "data"
# Date: 29/07/2024
# Author: Rushikesh Waghmare
#################################################
def GetData(Path):
    Data= pd.read_csv(Path)
    print("Top 5 entries of data: ",Data.head(5))
    return Data

#################################################
# Function_Name: ShowData()
# Discription: 
# Input: CSV file
# Output: Stores data in variable called "data"
# Date: 29/07/2024
# Author: Rushikesh Waghmare
#################################################
def ShowData(Data):
    print("Top 5 entries of data: ",Data.head(5))
    return Data

def Data_types(Data):
    print(Data.dtypes)
    print("\n")
    print(Data.info())

def DataInitialise(Data):
    X=Data.drop('GradeClass',axis=1)
    Y=Data['GradeClass']  
    return X,Y

def DataManipulation(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)
    return X_train, X_test, Y_train, Y_test 

# Visualization


def Visualisation(Data):
    #Age Histogram
    plt.hist(Data["Age"],bins=8,color="green")
    plt.title("Divided into Age")
    plt.xlabel("Count")
    plt.ylabel("Y axis")
    plt.show()

    #Gender
    plt.hist(Data["Gender"],bins=2,color="g")
    plt.title("Gender")
    plt.xlabel("Male and Female")
    plt.ylabel("count")
    plt.show()

    

    

# Model selection
def DecisionTree_Classifier(X_train, X_test, Y_train, Y_test):
    Dt=DecisionTreeClassifier(random_state=66)
    Dt.fit(X_train,Y_train)
    #print("Accuracy using DecisionTreeClassifier Algorithm: ",accuracy_score(X_train, Y_train)*100,"%")
    Y_pred=Dt.predict(X_test)
    print("Accuracy using DecisionTreeClassifier Algorithm {:.2f}%".format(accuracy_score(Y_test,Y_pred)*100))

def KNN_Classifier(X_train, X_test, Y_train, Y_test):
    Knn=KNeighborsClassifier(n_neighbors=3)
    Knn.fit(X_train,Y_train)
    #print("Accuracy using KNeighborsClassifier Algorithm: ",accuracy_score(X_train, Y_train)*100,"%")
    Y_pred=Knn.predict(X_test)
    print("Accuracy using KNeighborsClassifier Algorithm {:.2f}%".format(accuracy_score(Y_test,Y_pred)*100,"%"))

def LogisticAlgo_Classifier(X_train, X_test, Y_train, Y_test):
    LR=LogisticRegression(max_iter=1000)
    LR.fit(X_train,Y_train)
    #print("Accuracy using LogisticRegressionAlgorithm: ",accuracy_score(X_train, Y_train)*100,"%")
    Y_pred=LR.predict(X_test)
    print("Accuracy using LogisticRegression Algorithm {:.2f}%".format(accuracy_score(Y_test,Y_pred)*100,"%"))

def RandomForest_Classifier(X_train, X_test, Y_train, Y_test):
    RF=RandomForestClassifier(n_estimators=100)
    RF.fit(X_train,Y_train)
    #print("Accuracy using RandomForestClassifier Algorithm: ",accuracy_score(X_train, Y_train)*100,"%")
    Y_pred=RF.predict(X_test)
    print("Accuracy using RandomForestClassifier Algorithm {:.2f}%".format(accuracy_score(Y_test,Y_pred)*100,"%"))

def RandomForest_Classifier2(X_train, X_test, Y_train, Y_test):
    RF=RandomForestClassifier(max_depth = 3, n_estimators = 100,random_state = 0)
    RF.fit(X_train,Y_train)
    #print("Accuracy using RandomForestClassifier Algorithm: ",accuracy_score(X_train, Y_train)*100,"%")
    Y_pred=RF.predict(X_test)
    print("Accuracy using RandomForestClassifier Algorithm with Max Depth 3 is {:.2f}%".format(accuracy_score(Y_test,Y_pred)*100,"%"))

def LinearRegression_algorithm(X_train, X_test, Y_train, Y_test):
    LR=LinearRegression()
    LR.fit(X_train, Y_train)
    Y_pred=(X_test)
    print("Mean Squred error by using LinerRegression {:.2f}%".format(metrics.mean_squared_error(Y_test,Y_pred)))
#################################################
# Function_Name: main()
# Discription: Entry point function to call other functions
# Input: requred csv file
# Output: Output of given funtions
#################################################
def main():
    Data = GetData("Student_performance_data _.csv")
    #ShowData(Data)
    Data_types(Data)
    X,Y= DataInitialise(Data)
    X_train, X_test, Y_train, Y_test = DataManipulation(X,Y)
    Visualisation(Data)
    DecisionTree_Classifier(X_train, X_test, Y_train, Y_test )
   # KNN_Classifier(X_train, X_test, Y_train, Y_test )
    #LogisticAlgo_Classifier(X_train, X_test, Y_train, Y_test)
    RandomForest_Classifier(X_train, X_test, Y_train, Y_test)
    #RandomForest_Classifier2(X_train, X_test, Y_train, Y_test)
    #LinearRegression_algorithm(X_train, X_test, Y_train, Y_test)
if __name__ == "__main__":
    main()