import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("data.csv")

data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
data.head()
#malignat = M kotu huylu
#benign = B  iyi huylu


M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]


# scatter plot 
plt.scatter(M.radius_mean,M.texture_mean,color="red",label="Kotu")
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="iyi")
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

# knn modeli
knn = KNeighborsClassifier(n_neighbors = 8) #n_neighbors = k
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)

print(prediction)

print(" {} nn score : {} ".format(6,knn.score(x_test,y_test)))

#find k value
score_list = []
for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))

plt.plot(range(1,15),score_list)
plt.xlabel("k values")
plt.ylabel("score")
plt.show()

# yani k değerimiz 6 da en yüksek