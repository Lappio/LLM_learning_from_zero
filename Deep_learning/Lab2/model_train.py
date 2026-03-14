import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D


#读取训练集数据
data_train = 'Deep_learning\Lab2\data\train'
data_train= pathlib.Path(data_train)

data_val = 'Deep_learning\Lab2\data\val'
data_val = pathlib.Path(data_val)
#给数据类别放到列表数据中
CLASS_NAMES = np.array(['Cr','In','Pa','PS','Rs','Sc'])

# 设置图片大小和批次数
BATCH_SIZE = 64
IMG_HEIGHT= 32
IMG_WIDTH = 32

# 对数据进行归一化处理
image_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
# 训练集生成器
train_data_gen = image_generator.flow_from_directory(directory=str(data_train),batch_size=BATCH_SIZE,shuffle=True,target_size=(IMG_HEIGHT,IMG_WIDTH),classes=list(CLASS_NAMES)) 
# 读取路径 shuffle打乱图片
# 训练集生成器
val_data_gen = image_generator.flow_from_directory(directory=str(data_val),batch_size=BATCH_SIZE,shuffle=True,target_size=(IMG_HEIGHT,IMG_WIDTH),classes=list(CLASS_NAMES)) 