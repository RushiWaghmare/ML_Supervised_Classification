from sklearn import tree

def MarvellousClassifier(weight,surface):
    # Feature encoding
    Features = [[35, 1], [47, 1], [90, 0], [48, 1], [90, 1], [35, 1], [92, 0], [35, 1], [35, 1], [35, 1], [35, 1]]

    # Label Encoding
    Labels = [1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1]


    # Decide the algorithm
    obj = tree.DecisionTreeClassifier()

    # Train the algorithm
    obj =obj.fit(Features,Labels)

    #Test the model
    ret=obj.predict([[weight,surface]])
    if ret ==1:
        print("your ball looks like tennis ball")
    else:
        print("your ball looks like Cricket ball")

def main():
    print("-------------Ball type classification case study----------------")

    print("Please enter the information about the object that  you want to test")
    print("Please enter weight of your object in grams")
    no=int(input())
    print("please mention the type of surface Rough / Smooth")
    data=str(input())
    if data.lower() == "rough":
        data = 1
    elif data.lower() == "smooth":
        data = 0
    else:
        print("Invalid arguments")
        exit()

    MarvellousClassifier(no,data)


if __name__ == "__main__":
    main()