from cProfile import label
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("random+forest+regression+dataset.csv",sep = ";",header = None)
# print(df)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

# 

rf = RandomForestRegressor(n_estimators = 100, random_state = 42) #tree estimators, Random state 
rf.fit(x,y)

prediction = rf.predict([[5.8]])

print("5.8 seviyesinde fiyatın ne kadar olduğunu gösterir: ",prediction)

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = rf.predict(x_)

# visualize
plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("Tribun level")
plt.ylabel("Ücret")
plt.show()