import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #画图包
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#导入leras框架
import keras
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据预处理
dataset = pd.read_csv('breast_cancer_data.csv')
# print(dataset)
X = dataset.iloc[:,:-1]
# print(X)
Y = dataset.loc[ : ,'target']
# print(Y)
# 划分
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
# 转换为one-hot向量格式
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

#数据归一化
sc = MinMaxScaler(feature_range=(0,1))
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)




