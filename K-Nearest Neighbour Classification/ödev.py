import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("column_2C_weka.csv")

# print(data.head())

data = data.drop(["degree_spondylolisthesis"],axis =1)
print(data.head())

data["class"].values

data.info()

A = data[data["class"] == "Abnormal"]
N = data[data["class"] == "Normal"]


#abnormal = 1
#normal = 0

#görselleştirme
plt.scatter(A.pelvic_radius,A.sacral_slope,label= "Abnormal", color="red", alpha=0.4 )
plt.scatter(N.pelvic_radius,N.sacral_slope, label = "Normal", color="green", alpha=0.4)
plt.xlabel("pelvic_radius")
plt.ylabel("sacral_slope")
plt.legend()
plt.show()

data["class"] = [1 if each == "Abnormal" else 0 for each in data["class"]]
y = data["class"].values
print(y)

x_data = data.drop(["class"],axis=1)
print(x_data)

#normalization
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
print(x.head())

#train
x_train, x_test , y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)

#knn modeli
knn = KNeighborsClassifier(n_neighbors = 5) #k = 5
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print(prediction)


print("{} nn score : {} ".format(5,knn.score(x_test,y_test)))

##k değerinin en iyisini bulmamız gerekiyor
score_list = []
for k in range(1,20):
    knn2 = KNeighborsClassifier(n_neighbors = k)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    

#k görselleştirme
plt.plot(range(1,20),score_list)
plt.xlabel("k değeri")
plt.ylabel("score")

plt.show()

## en yüksek k değerim tabloda gösterildiği gibi k = 5 değerindedir..