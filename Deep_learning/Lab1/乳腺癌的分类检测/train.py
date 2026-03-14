import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #画图包
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#导入keras框架
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
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
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
# 转换为one-hot向量格式
y_train_one = to_categorical(y_train, 2)
y_test_one = to_categorical(y_test, 2)

#数据归一化
sc = MinMaxScaler(feature_range=(0,1))
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# 用Keras框架搭建深度学习网络模型
model = keras.Sequential()
model.add(Dense(10,activation='relu')) #Dense就是全连接网络模型
model.add(Dense(10,activation='relu'))
model.add(Dense(2,activation='softmax'))

# 对神经网络进行编译
model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])
#交叉熵 随机梯度下降法

# 模型训练

history = model.fit(x_train,y_train_one,epochs = 150,batch_size=16,verbose=2,validation_data=(x_test,y_test_one))
model.save('model.h5')

#绘制训练集和验证集的loss值对比
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='val')
plt.title("全连接神经网络loss值")
plt.legend()
plt.show()

#绘制训练集和验证集准确率对比图
plt.plot(history.history['accuracy'],label='train')
plt.plot(history.history['val_accuracy'],label='val')
plt.title("全连接神经网络loss值")
plt.legend()
plt.show()