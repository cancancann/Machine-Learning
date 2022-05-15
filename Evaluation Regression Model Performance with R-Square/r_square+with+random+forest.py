import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score #accurary

df = pd.read_csv("random+forest+regression+dataset (1).csv",sep = ";",header = None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

# 

rf = RandomForestRegressor(n_estimators = 300, random_state = 42)
rf.fit(x,y)

y_head = rf.predict(x)

# 
##estimators sayısını arttırınca accurary de artıyor..

print("r_score: %", r2_score(y,y_head) * 100 )