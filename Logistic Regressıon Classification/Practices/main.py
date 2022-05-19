import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")
datah = data.head()
# print(datah)
# datai = data.info()
datac = data.columns
# print(datac)

#1-> life
#0-> dead

#y yi bulacağız önce

y = data.target.values
# print(y)

x_data = data.drop(["target"],axis=1)
# print(x_data)

#Normalization
#Tüm verileri 0-1 dönüştürelim.

x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
# print(x)

##Train test split

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T




print("x_train:\n ",x_train.shape)
print("x_test: \n",x_test.shape)
print("y_train: \n",y_train.shape)
print("y_test: \n",y_test.shape)

##Parametre başlatma ve sigmoid işlevi

def weight_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0

    return w ,b 

def sigmoid(z):
    y_head = 1/(1 + np.exp(-z))
    
    return y_head

# print(sigmoid(0))#0.5


def forward_and_backward(w,b,x_train,y_train):
    
    #forward propagation
    z = np.dot(w.T,x_train) + b 
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head) # türev alma
    cost = (np.sum(loss)) / x_train.shape[1] #x_train.shape[1]= 952

    #backward propagation
    türev_w = (np.dot(x_train,((y_head-y_train).T))) / x_train.shape[1]
    türev_b = np.sum(y_head-y_train) / x_train.shape[1]
    sonuc = {"türev_w : ":türev_w, "türev_b : ": türev_b}

    return cost,sonuc


##guncelleme

def guncelleme(w,b,x_train,y_train,lr,noi):
    cost_list = []
    cost_list2 = []
    index  = []

    for i in range(noi):
        cost,sonuc = forward_and_backward(w,b,x_train,y_train)
        cost_list.append(i)

        w = w - lr * sonuc["türev_w"]
        b = b - lr * sonuc["türev_b"]

        if i % 10 == 0 :
            cost_list2.append(cost)
            index.append(i)
            print("cost after iteration: %i : %f" %(i,cost))
    # parametrelerin ağırlıklarını ve yanlılığını güncelliyoruz(öğreniyoruz)
    parameters = {"weight : ": w,"bias:":b}
    plt.plot(index,cost_list2)
    plt.xticks(index,cost_list2)
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()

    return parameters,sonuc,cost_list


def predict(w,b,x_test):
    z = sigmoid(np.dot(w.T,x_test)+ b)
    y_prediction = np.zeros((1,x_test.shape[1]))

    for i in range(z.shape[1]):
        if z[1]<= 0.5:
            y_prediction = 0
        else:
            y_prediction = 1

    return y_prediction

def logisticRegression(x_train,x_test,y_train,y_test,lr,noi):

    #başlangıç
    dimension = x_train.shape[0]
    w, b = weight_bias(dimension)

    parameters,sonuc,cost_list = guncelleme(w,b,x_train,y_train,lr,noi)

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)   

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))  


logisticRegression(x_train, y_train, x_test, y_test,lr=0.01, noi = 300)
