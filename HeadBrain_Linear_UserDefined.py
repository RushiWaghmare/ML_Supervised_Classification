######################################################
#User Defined Implimentation
######################################################

######################################################
# Import Requried Modules
######################################################
import pandas as pd
import numpy as np

######################################################
# Function_Name: 
# Discription: 
# Input: 
# Output: 
# Date: 29/07/2024
# Author: Rushikesh Waghmare
######################################################

######################################################
# Function_Name: Read_Data()
# Discription: Read Data from CSV file
# Input:  CSV file
# Output: Data
# Date: 29/07/2024
# Author: Rushikesh Waghmare
######################################################
def Read_Data(Path):
    Data=pd.read_csv(Path)
    print(Data.head(5))
    print(Data.info())
    #Data=Data.fillna(0)
    #print(Data.isnull().sum())
    return Data

######################################################
# Function_Name: Data_initialise()
# Discription: Divide Data into Dependet and independent Variabels
# Input: Data
# Output: X,Y
# Date: 29/07/2024
# Author: Rushikesh Waghmare
######################################################
def Data_initialise(Data):
    X = Data["Head Size(cm^3)"].values
    Y = Data["Brain Weight(grams)"].values
    return X,Y

######################################################
# Function_Name: Read_Data()
# Discription: Read Data from CSV file
# Input:  CSV file
# Output: Data
# Date: 29/07/2024
# Author: Rushikesh Waghmare
######################################################
def Mean_X_Y(X,Y):
    X_bar = np.mean(X)
    Y_bar = np.mean(Y)
    print(X_bar)
    print(Y_bar)

    # lenght of X
    N = len(X)
    return X_bar, Y_bar,N,X,Y

######################################################
# Function_Name: 
# Discription: 
# Input: 
# Output: 
# Date: 29/07/2024
# Author: Rushikesh Waghmare
######################################################
def Deviation(X_bar, Y_bar, N, X, Y):
    Sum_dev = 0
    Deno = 0
    for i in range(N):
        Sum_dev = Sum_dev + ((X[i]-X_bar)*(Y[i]-Y_bar))
    print("Sum of Product of Deviation of X and Y: ", Sum_dev)
    
    for i in range(N):
        Deno = Deno + (X[i]-X_bar)**2
    print(f"Sum of Square of Deviation of X: {Deno}")
    return Sum_dev , Deno,X_bar, Y_bar,N,X,Y

######################################################
# Function_Name: 
# Discription: 
# Input: 
# Output: 
# Date: 29/07/2024
# Author: Rushikesh Waghmare
######################################################
def Slope(Sum_dev , Deno,X_bar, Y_bar,N,X,Y):
    M=Sum_dev/Deno
    print("Slope of Line is ",M)
    return M,X_bar,Y_bar,N,X,Y

def Y_intersept(M,X_bar,Y_bar,N,X,Y):
    C = Y_bar -(M * X_bar)
    print(f"Value of C (Y Intercept) : {C}")
    return M,X_bar,Y_bar,N,C,X,Y

def PredictionOf_Y(M,X_bar,Y_bar,N,C,X,Y):
    Y_pred = []
    for i in range(N):
        Y_predi = (M *X[i]) + C
        Y_pred.append(Y_predi)
        #y_predi += Y_pred[]
    return Y_pred,M,X_bar,Y_bar,N,C,X,Y
 
def R_Square(Y_pred,M,X_bar,Y_bar,N,C,X,Y):
    Numerator = 0
    Denomenator = 0

    for i in range(N):

        Numerator = Numerator + ((Y_pred[i]-Y_bar)**2)
        Denomenator = Denomenator + ((Y[i]-Y_bar)**2)

    R_sq = Numerator / Denomenator

    print(f"Value of R_Square is : {R_sq}")

######################################################
# Function_Name: main()
# Discription: Entry point function
# Input: CSV path
# Output: 
# Date: 29/07/2024
# Author: Rushikesh Waghmare
######################################################
def main():
    Data = Read_Data("MarvellousHeadBrain.csv")

    X,Y = Data_initialise(Data)

    X_bar, Y_bar,N,X,Y = Mean_X_Y(X,Y)

    Sum_dev , Deno,X_bar, Y_bar,N,X,Y = Deviation(X_bar, Y_bar, N, X, Y)

    M,X_bar,Y_bar,N,X,Y = Slope(Sum_dev , Deno,X_bar, Y_bar,N,X,Y)

    M,X_bar,Y_bar,N,C,X,Y = Y_intersept(M,X_bar,Y_bar,N,X,Y)

    Y_pred,M,X_bar,Y_bar,N,C,X,Y = PredictionOf_Y(M,X_bar,Y_bar,N,C,X,Y)

    R_Square(Y_pred,M,X_bar,Y_bar,N,C,X,Y)

if __name__ == "__main__":
    main()