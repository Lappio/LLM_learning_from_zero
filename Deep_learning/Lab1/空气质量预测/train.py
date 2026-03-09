import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #画图包
import keras
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False






dataset = pd.read_csv("data.csv")
X = dataset.iloc[:,:-1]
Y = dataset['AQI']
# print(X)
# print(Y)
# 数据集的划分设置seed
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
# 归一化
sc_X = MinMaxScaler(feature_range=(0,1))
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
# print(y_test)
sc_Y = MinMaxScaler(feature_range=(0,1))
y_train = sc_Y.fit_transform(y_train)
y_test = sc_Y.transform(y_test)


