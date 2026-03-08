import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #画图包
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#导入leras框架
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from keras.models import load_model    #增加导入模型的包


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

# 导入训练好的模型
model = load_model("model.h5")

# 利用训练好的模型进行测试
predict = model.predict(x_test)
# print(predict)

y_pred = np.argmax(predict,axis=1)
# print(y_pred)
# 将结果保存为汉字
result = []
for i in range(len(y_pred)):
    if(y_pred[i] == 0):
        result.append("良性")
    else:
        result.append("恶性")

# 打印模型的精确度和召回
report = classification_report(y_test,y_pred,labels=[0,1],target_names=["良性","恶性"])
print(report)