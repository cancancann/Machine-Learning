from sklearn.tree import DecisionTreeRegressor ##tree
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("decision+tree+regression+dataset.csv",sep = ";",header = None)
# print(df)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)


#  decision tree regression

tree_reg = DecisionTreeRegressor()   # random state = 0
tree_reg.fit(x,y)


tree_reg.predict([[5.5]]) ## ([[]])
x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = tree_reg.predict(x_)

# visualize
plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color = "green")
plt.xlabel("Tribun Level")
plt.ylabel("Ücret")
plt.show()