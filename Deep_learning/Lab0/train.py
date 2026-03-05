import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  #用于划分数据集
from sklearn.preprocessing import MinMaxScaler #用于归一化
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#获取数据
dataset = pd.read_csv("breast_cancer_data.csv")  #确保代码文件路径与数据集路径在同一目录下
#读取数据集
#test
# print(dataset)
X = dataset.iloc[:, :-1] #iloc是pandas里面的用来取数据:,:先行后列
#test
# print(X)
Y  = dataset.loc[:,'target']    #loc用来取列
# print(Y)
#数据集的划分
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

#归一化处理
sc = MinMaxScaler(feature_range=(0,1))
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
# print(x_test)

#搭建模型并进行训练
lr = LogisticRegression()
lr.fit(x_train,y_train)
#用训练好的模型进行预测
pre_result = lr.predict(x_test)
# print(pre_result)
#预测的概率
pre_reslut_proba = lr.predict_proba(x_test)
# print(re_reslut_proba)
pre_list = pre_reslut_proba[:, 1] #获取第二列即恶性的概率
# 设置阈值
thresholds = 0.5
# 设置保存结果的列表
result = []
result_name = []
for i in range(len(pre_list)):
    if(pre_list[i]>thresholds):
        result.append(1)
        result_name.append('恶性')
    else:
        result.append(0)
        result_name.append('良性')
#模型评估
report = classification_report(y_test, result, labels=[0, 1], target_names=['良性肿瘤', '恶性肿瘤'])
print(report)