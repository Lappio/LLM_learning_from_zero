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
# 修改报错
y_train = sc_Y.fit_transform(y_train.to_frame())
y_test = sc_Y.transform(y_test.to_frame())

#···模型搭建

model = keras.Sequential()
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
# 回归任务输出一个值最后所得到的应该是一个值,不需要用激活函数
model.add(Dense(1))
# 模型编译
# SGD优化器,loss函数选择mes
model.compile(loss='mse',optimizer='SGD')
history = model.fit(x_train,y_train,epochs = 100,batch_size=16,verbose=2,validation_data=(x_test,y_test))
# verbose显示训练的过程
model.save("model.h5")

#绘制模型的训练集和验证集的loss值
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='val')
plt.title("全连接神经网络loss值")
plt.legend()
plt.show()