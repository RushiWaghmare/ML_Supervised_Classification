import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def GetData(Path):
    Data= pd.read_csv(Path)
    print(Data.head(5))
    print(Data.dtypes)
    print(Data.info())

    print("Size of Dataset: ",Data.shape)
    X=Data["Head Size(cm^3)"].values.reshape(-1,1)
    Y=Data["Brain Weight(grams)"].values


    reg=LinearRegression()
    reg=reg.fit(X,Y)

    y_pred = reg.predict(X)

    r2 = reg.score(X,Y)
    print(r2)

    mse = mean_squared_error(Y,y_pred)
    print(mse)





def main():
    Data=GetData("MarvellousHeadBrain.csv")
    
if __name__=="__main__":
    main()