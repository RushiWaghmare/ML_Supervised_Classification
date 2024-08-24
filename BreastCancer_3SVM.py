from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics


def MarvellousSVM():
    #Load dataset
    cancer = datasets.load_breast_cancer()

    #print the names of the 13 features
    print("Features of the cancer dataset: ", cancer.feature_names)

    #print  the label type of cancer('maligant' 'benign')
    print("Lables of the cancer dataset: ", cancer.target_names)

    #print data(feature)shape
    print("Labels of the cancer dataset: ",cancer.data.shape)

    #print data (features)shape
    print("shape of datset is : ",cancer.data.shape)

    #print the cancer data features(top 5 records)
    print("first 5 records are: ")
    print(cancer.data[0:5])

    # print the cancer labes (0:malignant,1:beingn)
    print("Traget of dataset: ",cancer.target)

    #split dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target,test_size=0.3, random_state =109)
    
    #Create a SVM Classifier
    clf = svm.SVC(kernel='linear')
    
    #Train the model using the training sets
    clf.fit(x_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(x_test)

    #Model Accuracy: how often is the classifier correct
    print("Accuracy of the model is : ",metrics.accuracy_score(y_test,y_pred)*100)

def main():
    print("______________Support Vecotor Machine_____________")

    MarvellousSVM()
if __name__ == "__main__":
    main()