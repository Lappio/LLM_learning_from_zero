#导入包
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt #画图包
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#导入keras框架
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from keras.models import load_model    #增加导入模型的包

#读取数据
train_data = pd.read_csv("/kaggle/input/competitions/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/competitions/titanic/test.csv")

#数据预处理
#异常处理_1 用中位数来填补年龄
train_data['Age'] = train_data.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))

# 1.定义想要提取的特征列（X 通常代表特征矩阵）
features = ['Pclass', 'Age', 'Parch', 'SibSp', 'Fare']

# 2. 提取 X_male：筛选出性别为 male 的行，并只保留 features 里的列
X_male = train_data[train_data['Sex'] == 'male'][features]
Y_male = train_data[train_data['Sex'] == 'male']['Survived']

# 3. 提取 X_female：筛选出性别为 female 的行，并只保留 features 里的列
X_female = train_data[train_data['Sex'] == 'female'][features]
Y_female = train_data[train_data['Sex'] == 'female']['Survived']
# print(Y_male)

# 对 X 进行归一化
sc_male = MinMaxScaler(feature_range=(0,1))
X_male = sc_male.fit_transform(X_male)
sc_female = MinMaxScaler(feature_range=(0,1))
X_female = sc_female.fit_transform(X_female)

# 用Keras框架搭建深度学习网络模型
model_male = keras.Sequential()
model_male.add(Dense(10,activation='relu')) #Dense就是全连接网络模型
model_male.add(Dense(10,activation='relu'))
model_male.add(Dense(2,activation='softmax'))

# 对神经网络进行编译
model_male.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])
#交叉熵 随机梯度下降法

# 模型训练

history = model_male.fit(X_male,Y_male,epochs = 150,batch_size=16,verbose=2)
model_male.save('model_male.h5')

# 用Keras框架搭建深度学习网络模型
model_female = keras.Sequential()
model_female.add(Dense(10,activation='relu')) #Dense就是全连接网络模型
model_female.add(Dense(10,activation='relu'))
model_female.add(Dense(2,activation='softmax'))

# 对神经网络进行编译
model_female.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])
#交叉熵 随机梯度下降法

# 模型训练

history = model_female.fit(X_female,Y_female,epochs = 150,batch_size=16,verbose=2)
model_female.save('model_female.h5')

# 检验模型

