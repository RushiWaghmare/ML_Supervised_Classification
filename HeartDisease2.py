import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,average_precision_score,f1_score

def Read_file(path):
    Data=pd.read_csv(path)
    print(Data.head(5))
    print(Data.shape)
    print(Data.dtypes)
    print(Data.info())
    return Data 

def Data_Initialise(Data):
    X=Data.drop("HeartDiseaseorAttack",axis=1)
    Y=Data["HeartDiseaseorAttack"]
    #print(X.head(5),Y.head(5))
    return X,Y 

def Train_Data(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=66)
    return X_train, X_test, Y_train, Y_test

def DecisionTree_Classifier(X_train, X_test, Y_train, Y_test):
    DT = DecisionTreeClassifier()
    DT.fit(X_train, Y_train)
    #print("Training Accuracy is :",accuracy_score(X_train, Y_train))
    Y_pred=DT.predict(X_test)
    print("Testing Accuracy is :",accuracy_score(Y_test,Y_pred))
    print("Confusion Matrics :",confusion_matrix(Y_test,Y_pred))
    print("AVG Precision Score :",average_precision_score(Y_test,Y_pred))
    print("F1 Score : ",f1_score(Y_test,Y_pred))
    print("="*35,end=" ")
    print("\n")
    

def Logistic_Reggression(X_train, X_test, Y_train, Y_test):
    LR = LogisticRegression(max_iter=400)
    LR.fit(X_train, Y_train)
    #print("Training Accuracy is :",accuracy_score(X_train, Y_train))
    Y_pred=LR.predict(X_test)
    print("Testing Accuracy is :",accuracy_score(Y_test,Y_pred))
    print("Confusion Matrics :",confusion_matrix(Y_test,Y_pred))
    print("AVG Precision Score :",average_precision_score(Y_test,Y_pred))
    print("F1 Score : ",f1_score(Y_test,Y_pred))
    print("="*35,end=" ")
    print("\n")

def main():
    Data=Read_file("HeartDisease2.csv")
    X,Y=Data_Initialise(Data)
    X_train, X_test, Y_train, Y_test=Train_Data(X,Y)
    DecisionTree_Classifier(X_train, X_test, Y_train, Y_test)
    Logistic_Reggression(X_train, X_test, Y_train, Y_test)
if __name__ == "__main__":
    main()