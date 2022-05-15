# import library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# import data
df = pd.read_csv("linear_regression_dataset.csv",sep = ";")

# plot data
plt.scatter(df.deneyim,df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()

# linear regression

# sklearn library

# linear regression model
linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

y_head = linear_reg.predict(x)  # maas

plt.plot(x, y_head,color = "red")


#
from sklearn.metrics import r2_score

print("r_square score: %", r2_score(y,y_head) * 100)


