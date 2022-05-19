import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("data.csv")

data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
data.head()

M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

# scatter plot 
plt.scatter(M.radius_mean,M.texture_mean,color="red",label="Kotu",alpha=0.4)
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="iyi",alpha=0.4)
plt.xlabel("radius")
plt.ylabel("texture")
plt.legend()
plt.show()

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)

# normalization
x = (x_data - np.min(x_data)) / (np.max(x_data)-np.min(x_data))

# train
x_train , x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)

#Naive Bayes
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train,y_train)

#score
print("accuracy of naive bayes algo : ",nb.score(x_test,y_test))

##accuracy = %93