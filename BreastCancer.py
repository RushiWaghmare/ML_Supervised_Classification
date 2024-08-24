import pandas as pd 
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

########################################################
"""
Function Name : ReadData():
Description : Read and load data from .csv file
Input : breast-cancer-wisconsin.csv file
Output : Data
Developer : Rushikesh Ratnakar Waghmare
Date : 05/07/2024
"""
########################################################
def load_data():

    data=pd.read_csv("breast-cancer-wisconsin.csv")
    #print(data.head(5))
    return data

########################################################
"""
Function Name : data_manipulation():
Description : clean the data
Input : Data
Output : Cleaned data
Developer : Rushikesh Ratnakar Waghmare
Date : 05/07/2024
"""
########################################################

def data_manipulation(data):
    #clean the data
    data.drop("CodeNumber",axis=1,inplace=True)

    # manupulate the data: repalce ? with NaN(Not a Data)
    data.replace('?',float('NaN'),inplace=True)
    #pd.to_numeric will convert 'Nan' with numeric values
    data=data.apply(pd.to_numeric,errors='coerce')
    #pd.fillna will replace values 'NaN' with 0
    data.fillna(0,inplace=True)
    print("Data after cleaning : ")
    print(data.head(5))
    return data

########################################################
"""
Function Name : select_data():
Description : Initialise the features and target
Input : cleaned data
Output : X,Y in form of features and target
Developer : Rushikesh Ratnakar Waghmare
Date : 05/07/2024
"""
########################################################
def select_data(data):

    X=data.drop("CancerType",axis=1)
    Y=data["CancerType"]
    return X,Y

########################################################
"""
Function Name :  split_data():
Description : split the data into train and test module
Input : Features and targets after initialisation
Output : splited data in test data and train data which is 30% and 70% per
Developer : Rushikesh Ratnakar Waghmare
Date : 05/07/2024
"""
########################################################
def split_data(X,Y):

    Xtrain,Xtest,Ytrain,Ytest= train_test_split(X,Y,test_size=0.3,random_state=42)
    return Xtrain,Xtest,Ytrain,Ytest

########################################################
"""
Function Name : decision_tree()
Description : Access the training and testing data , by using DecisionTressClassifier()
              train the algorithm and predict the data using the algorithm
Input :  splited data , module of neighbors
Output : predictions of data and accuracy of selected algorithm
Developer : Rushikesh Ratnakar Waghmare
Date : 05/07/2024
"""
########################################################
def decision_tree(Xtrain,Xtest,Ytrain,Ytest,data):

    classifier=tree.DecisionTreeClassifier()

    #Training Algorithm
    train=classifier.fit(Xtrain,Ytrain)
    
    #Test Algorithm
    y_pred=classifier.predict(Xtest)

    #Calculate Accuracy
    Accuracy=accuracy_score(Ytest,y_pred)
    print("Accuracy using Decision Tree Classifier Algorithm is : ",Accuracy*100,"%")

########################################################
"""
Function Name : decision_tree()
Description : Access the training and testing data , by using KNeighbors_Classifier()
              train the algorithm and predict the data using the algorithm
Input :  splited data , module of tree
Output : predictions of data and accuracy of selected algorithm
Developer : Rushikesh Ratnakar Waghmare
Date : 05/07/2024
"""
########################################################
def KNeighbors_Classifier(Xtrain,Xtest,Ytrain,Ytest,data):

    classifier=KNeighborsClassifier(n_neighbors=5)

    #Training Algorithm
    train=classifier.fit(Xtrain,Ytrain)
    
    #Test Algorithm
    y_pred=classifier.predict(Xtest)

    #Calculate Accuracy
    Accuracy=accuracy_score(Ytest,y_pred)
    print("Accuracy using KNeighors Classifier Algorithm is : ",Accuracy*100,"%")

########################################################
"""
Function Name : main():
Descriptio : used for running all above functions with required parameters
Input : run all functions
Output : output of above functions
Developer : Rushikesh Ratnakar Waghmare
Date : 05/07/2024
"""
########################################################
def main():

    data=load_data()
    clean=data_manipulation(data)
    X,Y=select_data(clean)
    Xtrain,Xtest,Ytrain,Ytest= split_data(X,Y)
    tree=decision_tree(Xtrain,Xtest,Ytrain,Ytest,clean)
    KNeighbors_Classifier(Xtrain,Xtest,Ytrain,Ytest,clean)


if __name__ == "__main__":
    main()