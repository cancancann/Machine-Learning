import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

## y = b0 + b1 * x1 +b2*x2...(feature sayısı artıyor..) algorithm

df = pd.read_csv("multiple_linear_regression_dataset.csv",sep=";")

# print(df)

x = df.iloc[:,[0,2]].values 
y = df.maas.values.reshape(-1,1)

#train fit
multiple_linear_reg = LinearRegression()
multiple_linear_reg.fit(x,y)

#predict
result = multiple_linear_reg.predict(np.array([[10,35],[5,35]]))
# print(result)

