import sklearn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

########################################################
"""
Function Name : ReadData():
Description : Read and load data from .csv file
Input : PlayPredict_File.csv file
Output : Data
Developer : Rushikesh Ratnakar Waghmare
Date : 04/07/2024
"""
########################################################
#Step 1: Read data
def ReadData():
    data=pd.read_csv("PlayPredict_File.csv")
    print("Top 5 entries from dataset: ")
    print(data.head(5))
    return data

########################################################
"""
Function Name : Encoding_data():
Description : Encoding the loaded data
Input : loaded data
Output : Encoded data
Developer : Rushikesh Ratnakar Waghmare
Date : 04/07/2024
"""
########################################################
#Step 2: Encoding Dataset
def Encoding_data(data):
    print("Data before Encoding")
    print(data.head(5))
    label_encoder=LabelEncoder()

    #Feature Encoding
    print("Data after Encoding ")
    data['Weather']=label_encoder.fit_transform(data['Weather'])
    data['Temprature']=label_encoder.fit_transform(data['Temprature'])
    print("Weather and Temprature columns after Encoding:")
    print(data['Weather'].head(5))
    print(data['Temprature'].head(5))
    
    #Target Encoding
    data["Play"]=label_encoder.fit_transform(data['Play'])
    print("Play column after Encoding:")
    print(data["Play"].head(5))
    return data

########################################################
"""
Function Name : Split_data():
Description : split the data in fetures and target
Input : encoded data
Output : splited data 50-50 % for training and testing
Developer : Rushikesh Ratnakar Waghmare
Date : 04/07/2024
"""
########################################################
#Step 3: Split the data
def Split_data(data):
    X=data[['Weather','Temprature']]
    Y=data['Play']

    Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.5)
    return Xtrain,Xtest,Ytrain,Ytest

########################################################
"""
Function Name : algorithms_selection():
Description : selection of suitable algorithm
Input : from sklearn.neighbors import KNeighborsClassifier
Output : Algorithm access
Developer : Rushikesh Ratnakar Waghmare
Date : 04/07/2024
"""
########################################################
#Step 4: Select Algorithms
def algorithms_selection():
    classifier=KNeighborsClassifier(n_neighbors=3)
    return classifier

########################################################
"""
Function Name : Train_algo()
Description : Training the algorithm with features
Input :  Xtrain and Ytrain
Output : Trained algorithm with 50 % data
Developer : Rushikesh Ratnakar Waghmare
Date : 04/07/2024
"""
########################################################
#Step 5: Train Algorithm
def Train_algo(Xtrain,Xtest,Ytrain,Ytest,classifier):
    train=classifier.fit(Xtrain,Ytrain)
    print("Model trained Successufully")
    return train
########################################################
"""
Function Name : Test_model():
Description : Testing the algorithm with some entries from remaining 50% testing data
Input : Testing data
Output : prediction of data
Developer : Rushikesh Ratnakar Waghmare
Date : 04/07/2024
"""
########################################################
#Step 6: Test Model
def Test_model(classifier):
    test=classifier.predict([[2,0]])
    if test=='1':
        print("Yes")
    else:
        print("No")
        
########################################################
"""
Function Name : main():
Descriptio : used for running all above functions with required parameters
Input : run all functions
Output : output of above functions
Developer : Rushikesh Ratnakar Waghmare
Date : 04/07/2024
"""
########################################################
def main():
    data=ReadData()
    data=Encoding_data(data)
    Xtrain,Xtest,Ytrain,Ytest = Split_data(data)
    classifier=algorithms_selection()
    trained_model=Train_algo(Xtrain,Xtest,Ytrain,Ytest,classifier)
    Testing=Test_model(classifier)

if __name__ == "__main__":
    main()