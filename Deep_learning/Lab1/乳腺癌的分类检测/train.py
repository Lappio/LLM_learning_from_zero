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

#解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
